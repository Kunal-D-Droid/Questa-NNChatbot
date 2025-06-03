import tensorflow as tf
import sys

def check_gpu():
    print("TensorFlow version:", tf.__version__)
    print("Python version:", sys.version)
    
    print("\nGPU Configuration:")
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"Found {len(gpus)} GPU(s):")
        for gpu in gpus:
            print(f"  - {gpu}")
        
        # Test GPU availability
        print("\nTesting GPU...")
        with tf.device('/GPU:0'):
            a = tf.random.normal([1000, 1000])
            b = tf.random.normal([1000, 1000])
            c = tf.matmul(a, b)
            print("GPU test successful!")
    else:
        print("No GPU found!")
    
    print("\nTensorFlow GPU Device:")
    print(tf.test.gpu_device_name())
    
    print("\nIs built with CUDA:", tf.test.is_built_with_cuda())
    print("Is built with GPU:", tf.test.is_built_with_gpu_support())

if __name__ == "__main__":
    check_gpu() 