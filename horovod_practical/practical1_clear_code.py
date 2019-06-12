import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import argparse

pools = 1
threads = 31

parser = argparse.ArgumentParser(description='First Horovod example.')
parser.add_argument('-datadir', type=str, default='/scratch-local' ,help='Folder containing mnist.npz')
parser.add_argument('-modeldir', type=str, default='modeldir' ,help='Folder where models are saved')
parser.add_argument('-batch_size', type=int, default=32, help='Batch size')
args = parser.parse_args()


def main(_): 
    os.environ['KMP_SETTINGS'] = str(1)
    os.environ['KMP_BLOCKTIME'] = str(0)
    os.environ['OMP_NUM_THREADS'] = str(threads)
    os.environ['KMP_AFFINITY'] = 'granularity=fine,compact,1,0'

    config = tf.ConfigProto()
    config.intra_op_parallelism_threads = threads
    config.inter_op_parallelism_threads = pools

    tf.enable_eager_execution(config=config)

    mnist_model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(16, [3, 3], activation='relu'),
        tf.keras.layers.Conv2D(16, [3, 3], activation='relu'),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(10)
    ])

    opt = tf.train.RMSPropOptimizer(0.001)

    (mnist_images, mnist_labels), _ = \
        tf.keras.datasets.mnist.load_data(path=os.path.join(args.datadir, 'mnist.npz'))

    dataset = tf.data.Dataset.from_tensor_slices(
        (tf.cast(mnist_images[..., tf.newaxis] / 255.0, tf.float32),
         tf.cast(mnist_labels, tf.int64))
    )
    dataset = dataset.shuffle(1000).batch(args.batch_size)

    checkpoint_dir = os.path.join(args.modeldir, 'checkpoints')
    step_counter = tf.train.get_or_create_global_step()
    print("Was here")
    checkpoint = tf.train.Checkpoint(model=mnist_model, optimizer=opt,
                                     step_counter=step_counter)

    for (batch, (images, labels)) in enumerate(dataset.take(2000)):
        with tf.GradientTape() as tape:
            logits = mnist_model(images, training=True)
            loss_value = tf.losses.sparse_softmax_cross_entropy(labels, logits)

        grads = tape.gradient(loss_value, mnist_model.variables)
        opt.apply_gradients(zip(grads, mnist_model.variables),
                            global_step=tf.train.get_or_create_global_step())

        if batch % 10 == 0:
            print('Step #%d\tLoss: %.6f' % (batch, loss_value))

    checkpoint.save(checkpoint_dir)

if __name__ == "__main__":
    tf.app.run()