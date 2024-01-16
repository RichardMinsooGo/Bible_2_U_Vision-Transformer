

! pip install patchify

! mkdir dataset

! wget https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz

!tar -xzvf "/content/flower_photos.tgz" -C "/content/dataset"

'''
Data Engineering
'''

'''
D1. Import Libraries for Data Engineering
'''

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


import tensorflow as tf
import numpy as np

import cv2
from glob import glob
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from patchify import patchify

""" Hyperparameters """
hyper_para = {}
hyper_para["image_size"]   = 200
hyper_para["num_channels"] = 3
hyper_para["patch_size"]   = 25
hyper_para["num_patches"]  = (hyper_para["image_size"]**2) // (hyper_para["patch_size"]**2)
hyper_para["flat_ptch_sh"] = (hyper_para["num_patches"], hyper_para["patch_size"]*hyper_para["patch_size"]*hyper_para["num_channels"])
# hyper_para["flat_patches_shape"] = (hyper_para["num_patches"], hyper_para["patch_size"]*hyper_para["patch_size"]*hyper_para["num_channels"])

hyper_para["batch_size"]   = 192
hyper_para["lr"]           = 1e-4
hyper_para["num_epochs"]   = 1  # 200
hyper_para["class_names"]  = ["daisy", "dandelion", "roses", "sunflowers", "tulips"]
hyper_para["num_classes"]  = len(hyper_para["class_names"])

hyper_para["num_layers"]   = 6        # 12
hyper_para["hidden_dim"]   = 384      # 768
hyper_para["mlp_dim"]      = 1024     # 3072
hyper_para["num_heads"]    = 8        # 12
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
    
    # print("Patchs Shape : ", patches)
    # print("Labels Shape : ", labels)
    
    labels = tf.one_hot(labels, hyper_para["num_classes"])

    patches.set_shape(hyper_para["flat_ptch_sh"])
    labels.set_shape(hyper_para["num_classes"])
    
    print("Patchs Shape : ", patches)
    print("Labels Shape : ", labels)
    
    return patches, labels

def tf_dataset(images, batch=32):
    ds = tf.data.Dataset.from_tensor_slices((images))
    ds = ds.map(parse).batch(batch).prefetch(8)
    return ds

'''
Model Engineering
'''

'''
M1. Import Libraries for Model Engineering
'''

from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras import optimizers
from tensorflow.keras import losses
from tensorflow.keras import metrics

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


"""
C02. Scaled dot product attention
"""
def ScaledDotProductAttention(query, key, value):
    """Calculate the attention weights.
    query, key, value must have matching leading dimensions.
    key, value must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
    The mask has different shapes depending on its type(padding or look ahead)
    but it must be broadcastable for addition.
    
    query, key, value의 leading dimensions은 동일해야 합니다.
    key, value 에는 일치하는 끝에서 두 번째 차원이 있어야 합니다(예: seq_len_k = seq_len_v).
    MASK는 유형에 따라 모양이 다릅니다(패딩 혹은 미리보기(=look ahead)).
    그러나 추가하려면 브로드캐스트할 수 있어야 합니다.

    Args:
        query: query shape == (batch_size, n_heads, seq_len_q, depth)
        key: key shape     == (batch_size, n_heads, seq_len_k, depth)
        value: value shape == (batch_size, n_heads, seq_len_v, depth_v)
        mask: Float tensor with shape broadcastable
              to (batch_size, n_heads, seq_len_q, seq_len_k). Defaults to None.

    Returns:
        output, attention_weights
    """
    
    # 1. MatMul Q, K-transpose. Attention score matrix.
    matmul_qk = tf.matmul(query, key, transpose_b=True)  # (..., seq_len_q, seq_len_k)

    # 2. scale matmul_qk
    dk = tf.cast(tf.shape(key)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    # 3. add the mask to the scaled tensor.
    # if mask is not None:
    #     scaled_attention_logits += (mask * -1e9)

    # 4. softmax is normalized on the last axis (seq_len_k) so that the scores add up to 1.
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)
    
    # 5. MatMul attn_prov, V
    output = tf.matmul(attention_weights, value)  # (..., seq_len_q, depth_v)

    return output, attention_weights

"""
C03. Multi head attention
"""
class MultiHeadAttentionLayer(tf.keras.layers.Layer):
    
    def __init__(self, hid_dim, n_heads):
        super(MultiHeadAttentionLayer, self).__init__()
        self.n_heads = n_heads
        assert hid_dim % self.n_heads == 0
        self.hid_dim = hid_dim
        
        # hid_dim divided by n_heads.
        self.depth = int(hid_dim/self.n_heads)
        
        # Define dense layers corresponding to WQ, WK, and WV
        self.q_linear = tf.keras.layers.Dense(hid_dim)
        self.k_linear = tf.keras.layers.Dense(hid_dim)
        self.v_linear = tf.keras.layers.Dense(hid_dim)
        
        # Dense layer definition corresponding to WO
        self.out = tf.keras.layers.Dense(hid_dim)

    def split_heads(self, inputs, batch_size):
        """Split the last dimension into (n_heads, depth).
        Transpose the result such that the shape is (batch_size, n_heads, seq_len, depth)
        """
        inputs = tf.reshape(
            inputs, (batch_size, -1, self.n_heads, self.depth))
        return tf.transpose(inputs, perm=[0, 2, 1, 3])

    def call(self, value, key, query):
        batch_size = tf.shape(query)[0]
        
        # 1. Pass through the dense layer corresponding to WQ
        # q : (batch_size, sentence length of query, hid_dim)
        query = self.q_linear(query)
        
        # split head
        # q : (batch_size, n_heads, sentence length of query, hid_dim/n_heads)
        query = self.split_heads(query, batch_size)
        
        # 2. Pass through the dense layer corresponding to WK
        # k : (batch_size, sentence length of key, hid_dim)
        key   = self.k_linear(key)
        
        # split head
        # k : (batch_size, n_heads, sentence length of key, hid_dim/n_heads)
        key   = self.split_heads(key, batch_size)
        
        # 3. Pass through the dense layer corresponding to WV
        # v : (batch_size, sentence length of value, hid_dim)
        value = self.v_linear(value)
        
        # split head
        # v : (batch_size, n_heads, sentence length of value, hid_dim/n_heads)
        value = self.split_heads(value, batch_size)
        
        # 4. Scaled Dot Product Attention. Using the previously implemented function
        # (batch_size, n_heads, sentence length of query, hid_dim/n_heads)
        # attention_weights.shape == (batch_size, n_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = ScaledDotProductAttention(
            query, key, value)
        
        # (batch_size, sentence length of query, n_heads, hid_dim/n_heads)
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        
        # 5. Concatenate the heads
        # (batch_size, sentence length of query, hid_dim)
        concat_attention = tf.reshape(scaled_attention,
                                      (batch_size, -1, self.hid_dim))
        
        # 6. Pass through the dense layer corresponding to WO
        # (batch_size, sentence length of query, hid_dim)
        outputs = self.out(concat_attention)

        return outputs

"""
C04. Positionwise Feedforward Layer
"""

def PositionwiseFeedforwardLayer(x, config):
    x = Dense(config["mlp_dim"], activation="gelu")(x)
    x = Dropout(config["dropout_rate"])(x)
    x = Dense(config["hidden_dim"])(x)
    x = Dropout(config["dropout_rate"])(x)
    return x


def transformer_encoder(x, config):
    skip_1 = x
    x = LayerNormalization()(x)
    
    x = MultiHeadAttentionLayer(config["hidden_dim"], config["num_heads"])(x, x, x)
    
    # print(x.shape)
    # x = MultiHeadAttention( num_heads=config["num_heads"], key_dim=config["hidden_dim"]  )(x, x)
    x = Add()([x, skip_1])

    skip_2 = x
    x = LayerNormalization()(x)
    x = PositionwiseFeedforwardLayer(x, config)
    x = Add()([x, skip_2])

    return x

def ViT(config):
    """ Inputs """
    input_shape = (config["num_patches"], config["patch_size"]*config["patch_size"]*config["num_channels"])
    inputs = Input(input_shape)     ## (None, 256, 3072)

    print("inputs Shape   : ", inputs.shape)

    """ Patch + Position Embeddings """
    patch_embed = Dense(config["hidden_dim"])(inputs)   ## (None, 256, 768)

    positions = tf.range(start=0, limit=config["num_patches"], delta=1)
    pos_embed = Embedding(input_dim=config["num_patches"], output_dim=config["hidden_dim"])(positions) ## (256, 768)
    embed     = patch_embed + pos_embed ## (None, 256, 768)

    print("Embedding Shape : ", embed.shape)
    
    """ Adding Class Token """
    token = ClassToken()(embed)
    print("Token Shape     : ", token.shape)
    
    x = Concatenate(axis=1)([token, embed]) ## (None, 257, 768)
    print("Concaten Shape  : ", x.shape)
    
    for _ in range(config["num_layers"]):
        x = transformer_encoder(x, config)

    """ Classification Head """
    x = LayerNormalization()(x)     ## (None, 257, 768)
    x = x[:, 0, :]
    x = Dense(config["num_classes"], activation="softmax")(x)

    print("Output Shape    : ", x.shape)
    
    model = Model(inputs, x)
    return model



# if __name__ == "__main__":
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

# --------------------------------------------------------------------------------------

optimizer=tf.keras.optimizers.Adam(hyper_para["lr"], clipvalue=1.0)

'''
M5. Define Loss Function
'''

criterion = losses.CategoricalCrossentropy()

train_loss = metrics.Mean(name='train_loss')
train_accuracy = metrics.CategoricalAccuracy(name='train_accuracy')

test_loss = metrics.Mean(name='test_loss')
test_accuracy = metrics.CategoricalAccuracy(name='test_accuracy')

'''
M6. Define train loop
'''

@tf.function
def train_step(images, labels):
    with tf.GradientTape() as tape:
        predictions = model(images)
        loss = criterion(labels, predictions)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    
    train_loss(loss)
    train_accuracy(labels, predictions)
    
'''
M7. Define validation / test loop
'''

@tf.function
def test_step(images, labels):
    predictions = model(images)
    t_loss = criterion(labels, predictions)
    
    test_loss(t_loss)
    test_accuracy(labels, predictions)

'''
M8. Define Episode / each step process
'''

from tqdm import tqdm, tqdm_notebook, trange

for epoch in range(hyper_para["num_epochs"]):
    
    with tqdm_notebook(total=len(train_ds), desc=f"Train Epoch {epoch+1}") as pbar:    
        train_losses = []
        train_accuracies = []
        
        for images, labels in train_ds:
            
            # print("images Shape : ", images)
            # print("labels Shape : ", labels)
            
            train_step(images, labels)
            
            loss_val= train_loss.result()
            acc     = train_accuracy.result()*100
            
            train_losses.append(loss_val)
            train_accuracies.append(acc)
            
            pbar.update(1)
            pbar.set_postfix_str(f"Loss: {loss_val:.4f} ({np.mean(train_losses):.4f}) Acc: {acc:.3f} ({np.mean(train_accuracies):.3f})")
            
'''
M9. Model evaluation
'''
with tqdm_notebook(total=len(valid_ds), desc=f"Test_ Epoch {epoch+1}") as pbar:    
    test_losses = []
    test_accuracies = []
    for test_images, test_labels in valid_ds:
        test_step(test_images, test_labels)

        loss_val= test_loss.result()
        acc     = test_accuracy.result()*100

        test_losses.append(loss_val)
        test_accuracies.append(acc)

        pbar.update(1)
        pbar.set_postfix_str(f"Loss: {loss_val:.4f} ({np.mean(test_losses):.4f}) Acc: {acc:.3f} ({np.mean(test_accuracies):.3f})")


"""
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
"""
    
    