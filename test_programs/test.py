import tensorflow as tf

# Check if TensorFlow is built with CUDA support
if tf.test.is_built_with_cuda():
    print("TensorFlow is built with CUDA support!")
else:
    print("TensorFlow is not built with CUDA support.")

# Check if CUDA GPU devices are available
gpu_devices = tf.config.list_physical_devices('GPU')
if len(gpu_devices) > 0:
    print("CUDA GPU devices are available!")
    for gpu in gpu_devices:
        print("Name:", gpu.name)
else:
    print("No CUDA GPU devices are available.")
