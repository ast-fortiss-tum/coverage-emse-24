import keras

class Singleton(object):
        def __new__(cls, *args, **kwds):
            it = cls.__dict__.get("__it__")
            if it is not None:
                return it
            cls.__it__ = it = object.__new__(cls)
            it.init(*args, **kwds)
            return it

        def init(self, *args, **kwds):
            pass

class MnistLoader(Singleton):
    def init(self):
        # load the MNIST dataset
        mnist = keras.datasets.mnist
        (_, _), (self.x_test, _) = mnist.load_data()

        print("++++++++++++++++++++ MnistLoader created")

    def get_x_test(self):
        return self.x_test

mnist_loader = MnistLoader()
