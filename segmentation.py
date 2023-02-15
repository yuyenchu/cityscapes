import tensorflow as tf
import tensorflow.keras.layers as layers
import numpy as np
import argparse
from datetime import datetime
from clearml import Task, Dataset
from utils import *

print(tf.__version__)

task = Task.init(project_name='semantic_segmentation', task_name='cityscapes segmentation')
configs = {'epochs': 500, 'batch_size': 12, 'base_lr': 0.001, 'model_type': 'enhanced_efm'}
configs = task.connect(configs) 
print(configs) 

dataset = Dataset.get(dataset_project='semantic_segmentation', dataset_name='cityscapes_fine', dataset_version='1.0.0-c416')
dataset_path = dataset.get_local_copy()

train_images = tf.data.Dataset.list_files(f'{dataset_path}/train/*/*_gtFine_labelIds.png').map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
test_images = tf.data.Dataset.list_files(f'{dataset_path}/val/*/*_gtFine_labelIds.png').map(load_image, num_parallel_calls=tf.data.AUTOTUNE)

EPOCHS = configs['epochs']
BATCH_SIZE = configs['batch_size']
VAL_SUBSPLITS = 5
BUFFER_SIZE = BATCH_SIZE*2
TRAIN_LENGTH = train_images.cardinality().numpy()
TEST_LENGTH = test_images.cardinality().numpy()
VALIDATION_STEPS = TEST_LENGTH//BATCH_SIZE//VAL_SUBSPLITS
STEPS_PER_EPOCH = TRAIN_LENGTH // BATCH_SIZE

train_batches = (
    train_images
    .shuffle(BUFFER_SIZE, seed=0)
    .batch(BATCH_SIZE)
    .repeat()
    .map(Augment())
    .prefetch(buffer_size=tf.data.AUTOTUNE))
test_batches = test_images.batch(BATCH_SIZE)

models = {
    'fpn': get_fpn,
    'enhanced_fpn': get_enhanced_fpn,
    'enhanced_efm': get_enhanced_efm,
    'enhanced_efm_small': get_enhanced_efm_small
}
model = models[configs['model_type']](8)
analyze(model)

# callbacks
logs = f'emf_enhanced{datetime.now().strftime("%Y%m%d-%H%M%S")}'
checkpoint_path = f'emf_enhanced{datetime.now().strftime("%Y%m%d-%H%M%S")}_ckpt'

tboard_callback = tf.keras.callbacks.TensorBoard(log_dir = logs,
                                                histogram_freq = 1,
                                                profile_batch = '200,220')
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 save_best_only=True,
                                                 verbose=1)
# model training
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=configs['base_lr']),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
            metrics=['accuracy'])

model_history = model.fit(train_batches, epochs=EPOCHS,
                        steps_per_epoch=STEPS_PER_EPOCH,
                        validation_steps=VALIDATION_STEPS,
                        validation_data=test_batches,
                        callbacks=[tboard_callback, cp_callback])
model.save('segmentation')
task.close()