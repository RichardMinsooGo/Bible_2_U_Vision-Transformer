

! pip install patchify

! mkdir dataset

! wget https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz

!tar -xzvf "/content/flower_photos.tgz" -C "/content/dataset"

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model

class ClassToken(Layer):
    def __init__(self):
        super().__init__()

    def build(self, input_shape):
        w_init = tf.random_normal_initializer()
        self.w = tf.Variable(
            initial_value = w_init(shape=(1, 1, input_shape[-1]), dtype=tf.float32),
            trainable = True
        )

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        hidden_dim = self.w.shape[-1]

        cls = tf.broadcast_to(self.w, [batch_size, 1, hidden_dim])
        cls = tf.cast(cls, dtype=inputs.dtype)
        return cls

def mlp(x, config):
    x = Dense(config["mlp_dim"], activation="gelu")(x)
    x = Dropout(config["dropout_rate"])(x)
    x = Dense(config["hidden_dim"])(x)
    x = Dropout(config["dropout_rate"])(x)
    return x

def transformer_encoder(x, config):
    skip_1 = x
    x = LayerNormalization()(x)
    x = MultiHeadAttention(
        num_heads=config["num_heads"], key_dim=config["hidden_dim"]
    )(x, x)
    x = Add()([x, skip_1])

    skip_2 = x
    x = LayerNormalization()(x)
    x = mlp(x, config)
    x = Add()([x, skip_2])

    return x

def ViT(config):
    """ Inputs """
    input_shape = (config["num_patches"], config["patch_size"]*config["patch_size"]*config["num_channels"])
    inputs = Input(input_shape)     ## (None, 256, 3072)

    """ Patch + Position Embeddings """
    patch_embed = Dense(config["hidden_dim"])(inputs)   ## (None, 256, 768)

    positions = tf.range(start=0, limit=config["num_patches"], delta=1)
    pos_embed = Embedding(input_dim=config["num_patches"], output_dim=config["hidden_dim"])(positions) ## (256, 768)
    embed     = patch_embed + pos_embed ## (None, 256, 768)

    """ Adding Class Token """
    token = ClassToken()(embed)
    x = Concatenate(axis=1)([token, embed]) ## (None, 257, 768)

    for _ in range(config["num_layers"]):
        x = transformer_encoder(x, config)

    """ Classification Head """
    x = LayerNormalization()(x)     ## (None, 257, 768)
    x = x[:, 0, :]
    x = Dense(config["num_classes"], activation="softmax")(x)

    model = Model(inputs, x)
    return model

import numpy as np
import cv2
from glob import glob
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from patchify import patchify
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau, EarlyStopping

""" Hyperparameters """
hyper_para = {}
hyper_para["image_size"]   = 200
hyper_para["num_channels"] = 3
hyper_para["patch_size"]   = 25
hyper_para["num_patches"]  = (hyper_para["image_size"]**2) // (hyper_para["patch_size"]**2)
hyper_para["flat_ptch_sh"] = (hyper_para["num_patches"], hyper_para["patch_size"]*hyper_para["patch_size"]*hyper_para["num_channels"])
# hyper_para["flat_patches_shape"] = (hyper_para["num_patches"], hyper_para["patch_size"]*hyper_para["patch_size"]*hyper_para["num_channels"])

hyper_para["batch_size"]   = 16
hyper_para["lr"]           = 1e-4
hyper_para["num_epochs"]   = 5  #200
hyper_para["num_classes"]  = 5
hyper_para["class_names"]  = ["daisy", "dandelion", "roses", "sunflowers", "tulips"]

hyper_para["num_layers"]   = 6 #12
hyper_para["hidden_dim"]   = 768
hyper_para["mlp_dim"]      = 3072
hyper_para["num_heads"]    = 12
hyper_para["dropout_rate"] = 0.1

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def load_data(path, split=0.1):
    images = shuffle(glob(os.path.join(path, "*", "*.jpg")))

    split_size = int(len(images) * split)
    train_x, valid_x = train_test_split(images, test_size=split_size, random_state=42)
    train_x, test_x  = train_test_split(train_x, test_size=split_size, random_state=42)

    return train_x, valid_x, test_x

def process_image_label(path):
    """ Reading images """
    path  = path.decode()
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    image = cv2.resize(image, (hyper_para["image_size"], hyper_para["image_size"]))
    image = image/255.0

    """ Preprocessing to patches """
    patch_shape = (hyper_para["patch_size"], hyper_para["patch_size"], hyper_para["num_channels"])
    patches = patchify(image, patch_shape, hyper_para["patch_size"])

    # patches = np.reshape(patches, (64, 25, 25, 3))
    # for i in range(64):
    #     cv2.imwrite(f"files/{i}.png", patches[i])

    patches = np.reshape(patches, hyper_para["flat_ptch_sh"])
    patches = patches.astype(np.float32)

    """ Label """
    class_name = path.split("/")[-2]
    class_idx = hyper_para["class_names"].index(class_name)
    class_idx = np.array(class_idx, dtype=np.int32)

    return patches, class_idx

def parse(path):
    patches, labels = tf.numpy_function(process_image_label, [path], [tf.float32, tf.int32])
    labels = tf.one_hot(labels, hyper_para["num_classes"])

    patches.set_shape(hyper_para["flat_ptch_sh"])
    labels.set_shape(hyper_para["num_classes"])

    return patches, labels

def tf_dataset(images, batch=32):
    ds = tf.data.Dataset.from_tensor_slices((images))
    ds = ds.map(parse).batch(batch).prefetch(8)
    return ds


if __name__ == "__main__":
    """ Seeding """
    np.random.seed(42)
    tf.random.set_seed(42)

    """ Directory for storing files """
    create_dir("files")

    """ Paths """
    dataset_path = "/content/dataset/flower_photos"
    model_path   = os.path.join("files", "model.h5")
    csv_path     = os.path.join("files", "log.csv")

    """ Dataset """
    train_x, valid_x, test_x = load_data(dataset_path)
    print(f"Train: {len(train_x)} - Valid: {len(valid_x)} - Test: {len(test_x)}")

    train_ds = tf_dataset(train_x, batch=hyper_para["batch_size"])
    valid_ds = tf_dataset(valid_x, batch=hyper_para["batch_size"])

    """ Model """
    model = ViT(hyper_para)
    model.compile(
        loss="categorical_crossentropy",
        optimizer=tf.keras.optimizers.Adam(hyper_para["lr"], clipvalue=1.0),
        metrics=["acc"]
    )

    callbacks = [
        ModelCheckpoint(model_path, monitor='val_loss', verbose=1, save_best_only=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, min_lr=1e-10, verbose=1),
        CSVLogger(csv_path),
        EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=False),
    ]

    model.fit(
        train_ds,
        epochs=hyper_para["num_epochs"],
        validation_data=valid_ds,
        callbacks=callbacks
    )
