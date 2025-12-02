from .nn import NN
from .. import activations
from .. import initializers
from .. import regularizers
from ... import config
from ...backend import tf
from ...utils import timing


class MIONet(NN):
    """Multiple-input operator network with two input functions."""

    def __init__(
        self,
        layer_sizes_branch1,
        layer_sizes_branch2,
        layer_sizes_trunk,
        activation,
        kernel_initializer,
        regularization=None,
    ):
        super().__init__()

        self.layer_branch1 = layer_sizes_branch1
        self.layer_branch2 = layer_sizes_branch2
        self.layer_trunk = layer_sizes_trunk
        if isinstance(activation, dict):
            self.activation_branch1 = activations.get(activation["branch1"])
            self.activation_branch2 = activations.get(activation["branch2"])
            self.activation_trunk = activations.get(activation["trunk"])
        else:
            self.activation_branch1 = (
                self.activation_branch2
            ) = self.activation_trunk = activations.get(activation)
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.regularizer = regularizers.get(regularization)

        self._inputs = None

    @property
    def inputs(self):
        return self._inputs

    @property
    def outputs(self):
        return self.y

    @property
    def targets(self):
        return self.target

    @timing
    def build(self):
        print("Building MIONet...")
        self.X_func1 = tf.placeholder(config.real(tf), [None, self.layer_branch1[0]])
        self.X_func2 = tf.placeholder(config.real(tf), [None, self.layer_branch2[0]])
        self.X_loc = tf.placeholder(config.real(tf), [None, self.layer_trunk[0]])
        self._inputs = [self.X_func1, self.X_func2, self.X_loc]

        # Branch net 1
        if callable(self.layer_branch1[1]):
            # User-defined network
            y_func1 = self.layer_branch1[1](self.X_func1)
        else:
            y_func1 = self._net(
                self.X_func1, self.layer_branch1[1:], self.activation_branch1
            )
        # Branch net 2
        if callable(self.layer_branch2[1]):
            # User-defined network
            y_func2 = self.layer_branch2[1](self.X_func2)
        else:
            y_func2 = self._net(
                self.X_func2, self.layer_branch2[1:], self.activation_branch2
            )
        # Trunk net
        y_loc = self._net(self.X_loc, self.layer_trunk[1:], self.activation_trunk)

        # Dot product
        self.y = tf.multiply(y_func1, y_loc)
        self.y = tf.multiply(self.y, y_func2)
        self.y = tf.reduce_sum(self.y, 1, keepdims=True)
        b = tf.Variable(tf.zeros(1))
        self.y += b

        self.target = tf.placeholder(config.real(tf), [None, 1])
        self.built = True

    def _net(self, X, layer, activation):
        output = X
        for i in range(len(layer) - 1):
            output = tf.layers.dense(
                output,
                layer[i],
                activation=activation,
                kernel_regularizer=self.regularizer,
            )
        return tf.layers.dense(output, layer[-1], kernel_regularizer=self.regularizer)


class MIONetCartesianProd(MIONet):
    """MIONet with two input functions for Cartesian product format."""

    @timing
    def build(self):
        print("Building MIONetCartesianProd...")

        self.X_func1 = tf.placeholder(config.real(tf), [None, self.layer_branch1[0]])
        self.X_func2 = tf.placeholder(config.real(tf), [None, self.layer_branch2[0]])
        self.X_loc = tf.placeholder(config.real(tf), [None, self.layer_trunk[0]])
        self._inputs = [self.X_func1, self.X_func2, self.X_loc]

        # Branch net 1
        if callable(self.layer_branch1[1]):
            # User-defined network
            y_func1 = self.layer_branch1[1](self.X_func1)
        else:
            y_func1 = self._net(
                self.X_func1, self.layer_branch1[1:], self.activation_branch1
            )
        # Branch net 2
        if callable(self.layer_branch2[1]):
            # User-defined network
            y_func2 = self.layer_branch2[1](self.X_func2)
        else:
            y_func2 = self._net(
                self.X_func2, self.layer_branch2[1:], self.activation_branch2
            )
        # Trunk net
        y_loc = self._net(self.X_loc, self.layer_trunk[1:], self.activation_trunk)

        # Dot product
        self.y = tf.multiply(y_func1, y_func2)
        self.y = tf.einsum("ip,jp->ij", self.y, y_loc)

        b = tf.Variable(tf.zeros(1))
        self.y += b
        self.target = tf.placeholder(config.real(tf), [None, None])
        self.built = True

### Customize MIONET for 3 branch net 
class MIONet_m3(NN):
    """Multiple-input operator network with 3 input functions."""

    def __init__(
        self,
        layer_sizes_branch1,
        layer_sizes_branch2,
        layer_sizes_branch3,
        layer_sizes_trunk,
        activation,
        kernel_initializer,
        regularization=None,
    ):
        super().__init__()

        self.layer_branch1 = layer_sizes_branch1
        self.layer_branch2 = layer_sizes_branch2
        self.layer_branch3 = layer_sizes_branch3
        self.layer_trunk = layer_sizes_trunk
        if isinstance(activation, dict):
            self.activation_branch1 = activations.get(activation["branch1"])
            self.activation_branch2 = activations.get(activation["branch2"])
            self.activation_branch3 = activations.get(activation["branch3"])
            self.activation_trunk = activations.get(activation["trunk"])
        else:
            self.activation_branch1 = (
                self.activation_branch2
            ) = self.activation_trunk = activations.get(activation)
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.regularizer = regularizers.get(regularization)

        self._inputs = None

    @property
    def inputs(self):
        return self._inputs

    @property
    def outputs(self):
        return self.y

    @property
    def targets(self):
        return self.target

    @timing
    def build(self):
        print("Building MIONet...")
        self.X_func1 = tf.placeholder(config.real(tf), [None, self.layer_branch1[0]])
        self.X_func2 = tf.placeholder(config.real(tf), [None, self.layer_branch2[0]])
        self.X_func3 = tf.placeholder(config.real(tf), [None, self.layer_branch3[0]])
        self.X_loc = tf.placeholder(config.real(tf), [None, self.layer_trunk[0]])
        self._inputs = [self.X_func1, self.X_func2, self.X_func3, self.X_loc]

        # Branch net 1
        if callable(self.layer_branch1[1]):
            # User-defined network
            y_func1 = self.layer_branch1[1](self.X_func1)
        else:
            y_func1 = self._net(
                self.X_func1, self.layer_branch1[1:], self.activation_branch1
            )
        # Branch net 2
        if callable(self.layer_branch2[1]):
            # User-defined network
            y_func2 = self.layer_branch2[1](self.X_func2)
        else:
            y_func2 = self._net(
                self.X_func2, self.layer_branch2[1:], self.activation_branch2
            )
        # Branch net 3
        if callable(self.layer_branch3[1]):
            # User-defined network
            y_func3 = self.layer_branch3[1](self.X_func3)
        else:
            y_func3 = self._net(
                self.X_func3, self.layer_branch3[1:], self.activation_branch3
            )
        # Trunk net
        y_loc = self._net(self.X_loc, self.layer_trunk[1:], self.activation_trunk)

        # Dot product
        self.y = tf.multiply(y_func1, y_loc)
        self.y = tf.multiply(self.y, y_func2)
        self.y = tf.multiply(self.y, y_func3)
        self.y = tf.reduce_sum(self.y, 1, keepdims=True)
        b = tf.Variable(tf.zeros(1))
        self.y += b

        self.target = tf.placeholder(config.real(tf), [None, 1])
        self.built = True

    def _net(self, X, layer, activation):
        output = X
        for i in range(len(layer) - 1):
            output = tf.layers.dense(
                output,
                layer[i],
                activation=activation,
                kernel_regularizer=self.regularizer,
            )
        return tf.layers.dense(output, layer[-1], kernel_regularizer=self.regularizer)

class MIONetCartesianProd_custom3(MIONet_m3):
    """MIONet with four input functions for Cartesian product format."""

    @timing
    def build(self):
        print("Building MIONetCartesianProd...")

        self.X_func1 = tf.placeholder(config.real(tf), [None, self.layer_branch1[0]])
        self.X_func2 = tf.placeholder(config.real(tf), [None, self.layer_branch2[0]])
        self.X_func3 = tf.placeholder(config.real(tf), [None, self.layer_branch3[0]])
        self.X_loc = tf.placeholder(config.real(tf), [None, self.layer_trunk[0]])
        self._inputs = [self.X_func1, self.X_func2, self.X_func3, self.X_loc]

        # Branch net 1
        if callable(self.layer_branch1[1]):
            # User-defined network
            y_func1 = self.layer_branch1[1](self.X_func1)
        else:
            y_func1 = self._net(
                self.X_func1, self.layer_branch1[1:], self.activation_branch1
            )
        # Branch net 2
        if callable(self.layer_branch2[1]):
            # User-defined network
            y_func2 = self.layer_branch2[1](self.X_func2)
        else:
            y_func2 = self._net(
                self.X_func2, self.layer_branch2[1:], self.activation_branch2
            )
        # Branch net 3
        if callable(self.layer_branch3[1]):
            # User-defined network
            y_func3 = self.layer_branch3[1](self.X_func3)
        else:
            y_func3 = self._net(
                self.X_func3, self.layer_branch3[1:], self.activation_branch3
            )
        # Trunk net
        y_loc = self._net(self.X_loc, self.layer_trunk[1:], self.activation_trunk)

        # Dot product
        self.y = tf.multiply(y_func1, y_func2)
        self.y = tf.multiply(self.y, y_func3)
        self.y = tf.einsum("ip,jp->ij", self.y, y_loc)

        b = tf.Variable(tf.zeros(1))
        self.y += b
        self.target = tf.placeholder(config.real(tf), [None, None])
        self.built = True

### Customize MIONET for 4 branch net 
class MIONet_m(NN):
    """Multiple-input operator network with 4 input functions."""

    def __init__(
        self,
        layer_sizes_branch1,
        layer_sizes_branch2,
        layer_sizes_branch3,
        layer_sizes_branch4,
        layer_sizes_trunk,
        activation,
        kernel_initializer,
        regularization=None,
    ):
        super().__init__()

        self.layer_branch1 = layer_sizes_branch1
        self.layer_branch2 = layer_sizes_branch2
        self.layer_branch3 = layer_sizes_branch3
        self.layer_branch4 = layer_sizes_branch4
        self.layer_trunk = layer_sizes_trunk
        if isinstance(activation, dict):
            self.activation_branch1 = activations.get(activation["branch1"])
            self.activation_branch2 = activations.get(activation["branch2"])
            self.activation_branch3 = activations.get(activation["branch3"])
            self.activation_branch4 = activations.get(activation["branch4"])
            self.activation_trunk = activations.get(activation["trunk"])
        else:
            self.activation_branch1 = (
                self.activation_branch2
            ) = self.activation_trunk = activations.get(activation)
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.regularizer = regularizers.get(regularization)

        self._inputs = None

    @property
    def inputs(self):
        return self._inputs

    @property
    def outputs(self):
        return self.y

    @property
    def targets(self):
        return self.target

    @timing
    def build(self):
        print("Building MIONet...")
        self.X_func1 = tf.placeholder(config.real(tf), [None, self.layer_branch1[0]])
        self.X_func2 = tf.placeholder(config.real(tf), [None, self.layer_branch2[0]])
        self.X_func3 = tf.placeholder(config.real(tf), [None, self.layer_branch3[0]])
        self.X_func4 = tf.placeholder(config.real(tf), [None, self.layer_branch4[0]])
        self.X_loc = tf.placeholder(config.real(tf), [None, self.layer_trunk[0]])
        self._inputs = [self.X_func1, self.X_func2, self.X_func3, self.X_func4, self.X_loc]

        # Branch net 1
        if callable(self.layer_branch1[1]):
            # User-defined network
            y_func1 = self.layer_branch1[1](self.X_func1)
        else:
            y_func1 = self._net(
                self.X_func1, self.layer_branch1[1:], self.activation_branch1
            )
        # Branch net 2
        if callable(self.layer_branch2[1]):
            # User-defined network
            y_func2 = self.layer_branch2[1](self.X_func2)
        else:
            y_func2 = self._net(
                self.X_func2, self.layer_branch2[1:], self.activation_branch2
            )
        # Branch net 3
        if callable(self.layer_branch3[1]):
            # User-defined network
            y_func3 = self.layer_branch3[1](self.X_func3)
        else:
            y_func3 = self._net(
                self.X_func3, self.layer_branch3[1:], self.activation_branch3
            )
        # Branch net 4
        if callable(self.layer_branch4[1]):
            # User-defined network
            y_func4 = self.layer_branch4[1](self.X_func4)
        else:
            y_func4 = self._net(
                self.X_func4, self.layer_branch4[1:], self.activation_branch4
            )
        # Trunk net
        y_loc = self._net(self.X_loc, self.layer_trunk[1:], self.activation_trunk)

        # Dot product
        self.y = tf.multiply(y_func1, y_loc)
        self.y = tf.multiply(self.y, y_func2)
        self.y = tf.multiply(self.y, y_func3)
        self.y = tf.multiply(self.y, y_func4)
        self.y = tf.reduce_sum(self.y, 1, keepdims=True)
        b = tf.Variable(tf.zeros(1))
        self.y += b

        self.target = tf.placeholder(config.real(tf), [None, 1])
        self.built = True

    def _net(self, X, layer, activation):
        output = X
        for i in range(len(layer) - 1):
            output = tf.layers.dense(
                output,
                layer[i],
                activation=activation,
                kernel_regularizer=self.regularizer,
            )
        return tf.layers.dense(output, layer[-1], kernel_regularizer=self.regularizer)

class MIONetCartesianProd_custom(MIONet_m):
    """MIONet with four input functions for Cartesian product format."""

    @timing
    def build(self):
        print("Building MIONetCartesianProd...")

        self.X_func1 = tf.placeholder(config.real(tf), [None, self.layer_branch1[0]])
        self.X_func2 = tf.placeholder(config.real(tf), [None, self.layer_branch2[0]])
        self.X_func3 = tf.placeholder(config.real(tf), [None, self.layer_branch3[0]])
        self.X_func4 = tf.placeholder(config.real(tf), [None, self.layer_branch4[0]])
        self.X_loc = tf.placeholder(config.real(tf), [None, self.layer_trunk[0]])
        self._inputs = [self.X_func1, self.X_func2, self.X_func3, self.X_func4, self.X_loc]

        # Branch net 1
        if callable(self.layer_branch1[1]):
            # User-defined network
            y_func1 = self.layer_branch1[1](self.X_func1)
        else:
            y_func1 = self._net(
                self.X_func1, self.layer_branch1[1:], self.activation_branch1
            )
        # Branch net 2
        if callable(self.layer_branch2[1]):
            # User-defined network
            y_func2 = self.layer_branch2[1](self.X_func2)
        else:
            y_func2 = self._net(
                self.X_func2, self.layer_branch2[1:], self.activation_branch2
            )
        # Branch net 3
        if callable(self.layer_branch3[1]):
            # User-defined network
            y_func3 = self.layer_branch3[1](self.X_func3)
        else:
            y_func3 = self._net(
                self.X_func3, self.layer_branch3[1:], self.activation_branch3
            )
        # Branch net 4
        if callable(self.layer_branch4[1]):
            # User-defined network
            y_func4 = self.layer_branch4[1](self.X_func4)
        else:
            y_func4 = self._net(
                self.X_func4, self.layer_branch4[1:], self.activation_branch4
            )
        # Trunk net
        y_loc = self._net(self.X_loc, self.layer_trunk[1:], self.activation_trunk)

        # Dot product
        self.y = tf.multiply(y_func1, y_func2)
        self.y = tf.multiply(self.y, y_func3)
        self.y = tf.multiply(self.y, y_func4)
        self.y = tf.einsum("ip,jp->ij", self.y, y_loc)

        b = tf.Variable(tf.zeros(1))
        self.y += b
        self.target = tf.placeholder(config.real(tf), [None, None])
        self.built = True

## 8 branchnet for SMR
class MIONet_m8(NN):
    """Multiple-input operator network with 8 input functions."""

    def __init__(
        self,
        layer_sizes_branch1,
        layer_sizes_branch2,
        layer_sizes_branch3,
        layer_sizes_branch4,
        layer_sizes_branch5,
        layer_sizes_branch6,
        layer_sizes_branch7,
        layer_sizes_branch8,
        layer_sizes_trunk,
        activation,
        kernel_initializer,
        regularization=None,
    ):
        super().__init__()

        self.layer_branch1 = layer_sizes_branch1
        self.layer_branch2 = layer_sizes_branch2
        self.layer_branch3 = layer_sizes_branch3
        self.layer_branch4 = layer_sizes_branch4
        self.layer_branch5 = layer_sizes_branch5
        self.layer_branch6 = layer_sizes_branch6
        self.layer_branch7 = layer_sizes_branch7
        self.layer_branch8 = layer_sizes_branch8
        self.layer_trunk = layer_sizes_trunk
        if isinstance(activation, dict):
            self.activation_branch1 = activations.get(activation["branch1"])
            self.activation_branch2 = activations.get(activation["branch2"])
            self.activation_branch3 = activations.get(activation["branch3"])
            self.activation_branch4 = activations.get(activation["branch4"])
            self.activation_branch5 = activations.get(activation["branch5"])
            self.activation_branch6 = activations.get(activation["branch6"])
            self.activation_branch7 = activations.get(activation["branch7"])
            self.activation_branch8 = activations.get(activation["branch8"])
            self.activation_trunk = activations.get(activation["trunk"])
        else:
            self.activation_branch1 = (
                self.activation_branch2
            )= (
                self.activation_branch3
            )= (
                self.activation_branch4
            )= (
                self.activation_branch5
            )= (
                self.activation_branch6
            )= (
                self.activation_branch7
            )= (
                self.activation_branch8
            ) = self.activation_trunk = activations.get(activation)
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.regularizer = regularizers.get(regularization)

        self._inputs = None

    @property
    def inputs(self):
        return self._inputs

    @property
    def outputs(self):
        return self.y

    @property
    def targets(self):
        return self.target

    @timing
    def build(self):
        print("Building 8 branchs MIONet...")
        self.X_func1 = tf.placeholder(config.real(tf), [None, self.layer_branch1[0]])
        self.X_func2 = tf.placeholder(config.real(tf), [None, self.layer_branch2[0]])
        self.X_func3 = tf.placeholder(config.real(tf), [None, self.layer_branch3[0]])
        self.X_func4 = tf.placeholder(config.real(tf), [None, self.layer_branch4[0]])
        self.X_func5 = tf.placeholder(config.real(tf), [None, self.layer_branch5[0]])
        self.X_func6 = tf.placeholder(config.real(tf), [None, self.layer_branch6[0]])
        self.X_func7 = tf.placeholder(config.real(tf), [None, self.layer_branch7[0]])
        self.X_func8 = tf.placeholder(config.real(tf), [None, self.layer_branch8[0]])
        self.X_loc = tf.placeholder(config.real(tf), [None, self.layer_trunk[0]])
        self._inputs = [self.X_func1, self.X_func2, self.X_func3, self.X_func4,
                        self.X_func5, self.X_func6, self.X_func7, self.X_func8, self.X_loc]

        # Branch net 1
        if callable(self.layer_branch1[1]):
            # User-defined network
            y_func1 = self.layer_branch1[1](self.X_func1)
        else:
            y_func1 = self._net(
                self.X_func1, self.layer_branch1[1:], self.activation_branch1
            )
        # Branch net 2
        if callable(self.layer_branch2[1]):
            # User-defined network
            y_func2 = self.layer_branch2[1](self.X_func2)
        else:
            y_func2 = self._net(
                self.X_func2, self.layer_branch2[1:], self.activation_branch2
            )
        # Branch net 3
        if callable(self.layer_branch3[1]):
            # User-defined network
            y_func3 = self.layer_branch3[1](self.X_func3)
        else:
            y_func3 = self._net(
                self.X_func3, self.layer_branch3[1:], self.activation_branch3
            )
        # Branch net 4
        if callable(self.layer_branch4[1]):
            # User-defined network
            y_func4 = self.layer_branch4[1](self.X_func4)
        else:
            y_func4 = self._net(
                self.X_func4, self.layer_branch4[1:], self.activation_branch4
            )
        # Branch net 5
        if callable(self.layer_branch5[1]):
            # User-defined network
            y_func5 = self.layer_branch5[1](self.X_func5)
        else:
            y_func5 = self._net(
                self.X_func5, self.layer_branch5[1:], self.activation_branch5
            )
        # Branch net 6
        if callable(self.layer_branch6[1]):
            # User-defined network
            y_func6 = self.layer_branch6[1](self.X_func6)
        else:
            y_func6 = self._net(
                self.X_func6, self.layer_branch6[1:], self.activation_branch6
            )
        # Branch net 7
        if callable(self.layer_branch7[1]):
            # User-defined network
            y_func7 = self.layer_branch7[1](self.X_func7)
        else:
            y_func7 = self._net(
                self.X_func7, self.layer_branch7[1:], self.activation_branch7
            )
        # Branch net 8
        if callable(self.layer_branch8[1]):
            # User-defined network
            y_func8 = self.layer_branch8[1](self.X_func8)
        else:
            y_func8 = self._net(
                self.X_func8, self.layer_branch8[1:], self.activation_branch8
            )
        # Trunk net
        y_loc = self._net(self.X_loc, self.layer_trunk[1:], self.activation_trunk)

        # Dot product
        self.y = tf.multiply(y_func1, y_loc)
        self.y = tf.multiply(self.y, y_func2)
        self.y = tf.multiply(self.y, y_func3)
        self.y = tf.multiply(self.y, y_func4)
        self.y = tf.multiply(self.y, y_func5)
        self.y = tf.multiply(self.y, y_func6)
        self.y = tf.multiply(self.y, y_func7)
        self.y = tf.multiply(self.y, y_func8)
        self.y = tf.reduce_sum(self.y, 1, keepdims=True)
        b = tf.Variable(tf.zeros(1))
        self.y += b

        self.target = tf.placeholder(config.real(tf), [None, 1])
        self.built = True

    def _net(self, X, layer, activation):
        output = X
        for i in range(len(layer) - 1):
            output = tf.layers.dense(
                output,
                layer[i],
                activation=activation,
                kernel_regularizer=self.regularizer,
            )
        return tf.layers.dense(output, layer[-1], kernel_regularizer=self.regularizer)

class MIONetCartesianProd_custom8(MIONet_m8):
    """MIONet with four input functions for Cartesian product format for 8 branch net."""

    @timing
    def build(self):
        print("Building MIONetCartesianProd...")

        self.X_func1 = tf.placeholder(config.real(tf), [None, self.layer_branch1[0]])
        self.X_func2 = tf.placeholder(config.real(tf), [None, self.layer_branch2[0]])
        self.X_func3 = tf.placeholder(config.real(tf), [None, self.layer_branch3[0]])
        self.X_func4 = tf.placeholder(config.real(tf), [None, self.layer_branch4[0]])
        self.X_func5 = tf.placeholder(config.real(tf), [None, self.layer_branch5[0]])
        self.X_func6 = tf.placeholder(config.real(tf), [None, self.layer_branch6[0]])
        self.X_func7 = tf.placeholder(config.real(tf), [None, self.layer_branch7[0]])
        self.X_func8 = tf.placeholder(config.real(tf), [None, self.layer_branch8[0]])
        self.X_loc = tf.placeholder(config.real(tf), [None, self.layer_trunk[0]])
        self._inputs = [self.X_func1, self.X_func2, self.X_func3, self.X_func4, 
                        self.X_func5, self.X_func6, self.X_func7, self.X_func8, self.X_loc]

        # Branch net 1
        if callable(self.layer_branch1[1]):
            # User-defined network
            y_func1 = self.layer_branch1[1](self.X_func1)
        else:
            y_func1 = self._net(
                self.X_func1, self.layer_branch1[1:], self.activation_branch1
            )
        # Branch net 2
        if callable(self.layer_branch2[1]):
            # User-defined network
            y_func2 = self.layer_branch2[1](self.X_func2)
        else:
            y_func2 = self._net(
                self.X_func2, self.layer_branch2[1:], self.activation_branch2
            )
        # Branch net 3
        if callable(self.layer_branch3[1]):
            # User-defined network
            y_func3 = self.layer_branch3[1](self.X_func3)
        else:
            y_func3 = self._net(
                self.X_func3, self.layer_branch3[1:], self.activation_branch3
            )
        # Branch net 4
        if callable(self.layer_branch4[1]):
            # User-defined network
            y_func4 = self.layer_branch4[1](self.X_func4)
        else:
            y_func4 = self._net(
                self.X_func4, self.layer_branch4[1:], self.activation_branch4
            )
        # Branch net 5
        if callable(self.layer_branch5[1]):
            # User-defined network
            y_func5 = self.layer_branch5[1](self.X_func5)
        else:
            y_func5 = self._net(
                self.X_func5, self.layer_branch5[1:], self.activation_branch5
            )
        # Branch net 6
        if callable(self.layer_branch6[1]):
            # User-defined network
            y_func6 = self.layer_branch6[1](self.X_func6)
        else:
            y_func6 = self._net(
                self.X_func6, self.layer_branch6[1:], self.activation_branch6
            )
        # Branch net 7
        if callable(self.layer_branch7[1]):
            # User-defined network
            y_func7 = self.layer_branch7[1](self.X_func7)
        else:
            y_func7 = self._net(
                self.X_func7, self.layer_branch7[1:], self.activation_branch7
            )
        # Branch net 8
        if callable(self.layer_branch8[1]):
            # User-defined network
            y_func8 = self.layer_branch8[1](self.X_func8)
        else:
            y_func8 = self._net(
                self.X_func8, self.layer_branch8[1:], self.activation_branch8
            )
        # Trunk net
        y_loc = self._net(self.X_loc, self.layer_trunk[1:], self.activation_trunk)

        # Dot product
        self.y = tf.multiply(y_func1, y_func2)
        self.y = tf.multiply(self.y, y_func3)
        self.y = tf.multiply(self.y, y_func4)
        self.y = tf.multiply(self.y, y_func5)
        self.y = tf.multiply(self.y, y_func6)
        self.y = tf.multiply(self.y, y_func7)
        self.y = tf.multiply(self.y, y_func8)
        self.y = tf.einsum("ip,jp->ij", self.y, y_loc)

        b = tf.Variable(tf.zeros(1))
        self.y += b
        self.target = tf.placeholder(config.real(tf), [None, None])
        self.built = True

## 7 branchnet for PWR
class MIONet_m7(NN):
    """Multiple-input operator network with 7 input functions."""

    def __init__(
        self,
        layer_sizes_branch1,
        layer_sizes_branch2,
        layer_sizes_branch3,
        layer_sizes_branch4,
        layer_sizes_branch5,
        layer_sizes_branch6,
        layer_sizes_branch7,
        layer_sizes_trunk,
        activation,
        kernel_initializer,
        regularization=None,
    ):
        super().__init__()

        self.layer_branch1 = layer_sizes_branch1
        self.layer_branch2 = layer_sizes_branch2
        self.layer_branch3 = layer_sizes_branch3
        self.layer_branch4 = layer_sizes_branch4
        self.layer_branch5 = layer_sizes_branch5
        self.layer_branch6 = layer_sizes_branch6
        self.layer_branch7 = layer_sizes_branch7
        self.layer_trunk = layer_sizes_trunk
        if isinstance(activation, dict):
            self.activation_branch1 = activations.get(activation["branch1"])
            self.activation_branch2 = activations.get(activation["branch2"])
            self.activation_branch3 = activations.get(activation["branch3"])
            self.activation_branch4 = activations.get(activation["branch4"])
            self.activation_branch5 = activations.get(activation["branch5"])
            self.activation_branch6 = activations.get(activation["branch6"])
            self.activation_branch7 = activations.get(activation["branch7"])
            self.activation_trunk = activations.get(activation["trunk"])
        else:
            self.activation_branch1 = (
                self.activation_branch2
            )= (
                self.activation_branch3
            )= (
                self.activation_branch4
            )= (
                self.activation_branch5
            )= (
                self.activation_branch6
            )= (
                self.activation_branch7
            )= self.activation_trunk = activations.get(activation)
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.regularizer = regularizers.get(regularization)

        self._inputs = None

    @property
    def inputs(self):
        return self._inputs

    @property
    def outputs(self):
        return self.y

    @property
    def targets(self):
        return self.target

    @timing
    def build(self):
        print("Building 7 branchs MIONet...")
        self.X_func1 = tf.placeholder(config.real(tf), [None, self.layer_branch1[0]])
        self.X_func2 = tf.placeholder(config.real(tf), [None, self.layer_branch2[0]])
        self.X_func3 = tf.placeholder(config.real(tf), [None, self.layer_branch3[0]])
        self.X_func4 = tf.placeholder(config.real(tf), [None, self.layer_branch4[0]])
        self.X_func5 = tf.placeholder(config.real(tf), [None, self.layer_branch5[0]])
        self.X_func6 = tf.placeholder(config.real(tf), [None, self.layer_branch6[0]])
        self.X_func7 = tf.placeholder(config.real(tf), [None, self.layer_branch7[0]])
        self.X_loc = tf.placeholder(config.real(tf), [None, self.layer_trunk[0]])
        self._inputs = [self.X_func1, self.X_func2, self.X_func3, self.X_func4,
                        self.X_func5, self.X_func6, self.X_func7, self.X_loc]

        # Branch net 1
        if callable(self.layer_branch1[1]):
            # User-defined network
            y_func1 = self.layer_branch1[1](self.X_func1)
        else:
            y_func1 = self._net(
                self.X_func1, self.layer_branch1[1:], self.activation_branch1
            )
        # Branch net 2
        if callable(self.layer_branch2[1]):
            # User-defined network
            y_func2 = self.layer_branch2[1](self.X_func2)
        else:
            y_func2 = self._net(
                self.X_func2, self.layer_branch2[1:], self.activation_branch2
            )
        # Branch net 3
        if callable(self.layer_branch3[1]):
            # User-defined network
            y_func3 = self.layer_branch3[1](self.X_func3)
        else:
            y_func3 = self._net(
                self.X_func3, self.layer_branch3[1:], self.activation_branch3
            )
        # Branch net 4
        if callable(self.layer_branch4[1]):
            # User-defined network
            y_func4 = self.layer_branch4[1](self.X_func4)
        else:
            y_func4 = self._net(
                self.X_func4, self.layer_branch4[1:], self.activation_branch4
            )
        # Branch net 5
        if callable(self.layer_branch5[1]):
            # User-defined network
            y_func5 = self.layer_branch5[1](self.X_func5)
        else:
            y_func5 = self._net(
                self.X_func5, self.layer_branch5[1:], self.activation_branch5
            )
        # Branch net 6
        if callable(self.layer_branch6[1]):
            # User-defined network
            y_func6 = self.layer_branch6[1](self.X_func6)
        else:
            y_func6 = self._net(
                self.X_func6, self.layer_branch6[1:], self.activation_branch6
            )
        # Branch net 7
        if callable(self.layer_branch7[1]):
            # User-defined network
            y_func7 = self.layer_branch7[1](self.X_func7)
        else:
            y_func7 = self._net(
                self.X_func7, self.layer_branch7[1:], self.activation_branch7
            )
        # Trunk net
        y_loc = self._net(self.X_loc, self.layer_trunk[1:], self.activation_trunk)

        # Dot product
        self.y = tf.multiply(y_func1, y_loc)
        self.y = tf.multiply(self.y, y_func2)
        self.y = tf.multiply(self.y, y_func3)
        self.y = tf.multiply(self.y, y_func4)
        self.y = tf.multiply(self.y, y_func5)
        self.y = tf.multiply(self.y, y_func6)
        self.y = tf.multiply(self.y, y_func7)
        self.y = tf.reduce_sum(self.y, 1, keepdims=True)
        b = tf.Variable(tf.zeros(1))
        self.y += b

        self.target = tf.placeholder(config.real(tf), [None, 1])
        self.built = True

    def _net(self, X, layer, activation):
        output = X
        for i in range(len(layer) - 1):
            output = tf.layers.dense(
                output,
                layer[i],
                activation=activation,
                kernel_regularizer=self.regularizer,
            )
        return tf.layers.dense(output, layer[-1], kernel_regularizer=self.regularizer)

class MIONetCartesianProd_custom7(MIONet_m7):
    """MIONet with four input functions for Cartesian product format for 7 branch net."""

    @timing
    def build(self):
        print("Building MIONetCartesianProd...")

        self.X_func1 = tf.placeholder(config.real(tf), [None, self.layer_branch1[0]])
        self.X_func2 = tf.placeholder(config.real(tf), [None, self.layer_branch2[0]])
        self.X_func3 = tf.placeholder(config.real(tf), [None, self.layer_branch3[0]])
        self.X_func4 = tf.placeholder(config.real(tf), [None, self.layer_branch4[0]])
        self.X_func5 = tf.placeholder(config.real(tf), [None, self.layer_branch5[0]])
        self.X_func6 = tf.placeholder(config.real(tf), [None, self.layer_branch6[0]])
        self.X_func7 = tf.placeholder(config.real(tf), [None, self.layer_branch7[0]])
        self.X_loc = tf.placeholder(config.real(tf), [None, self.layer_trunk[0]])
        self._inputs = [self.X_func1, self.X_func2, self.X_func3, self.X_func4, 
                        self.X_func5, self.X_func6, self.X_func7, self.X_loc]

        # Branch net 1
        if callable(self.layer_branch1[1]):
            # User-defined network
            y_func1 = self.layer_branch1[1](self.X_func1)
        else:
            y_func1 = self._net(
                self.X_func1, self.layer_branch1[1:], self.activation_branch1
            )
        # Branch net 2
        if callable(self.layer_branch2[1]):
            # User-defined network
            y_func2 = self.layer_branch2[1](self.X_func2)
        else:
            y_func2 = self._net(
                self.X_func2, self.layer_branch2[1:], self.activation_branch2
            )
        # Branch net 3
        if callable(self.layer_branch3[1]):
            # User-defined network
            y_func3 = self.layer_branch3[1](self.X_func3)
        else:
            y_func3 = self._net(
                self.X_func3, self.layer_branch3[1:], self.activation_branch3
            )
        # Branch net 4
        if callable(self.layer_branch4[1]):
            # User-defined network
            y_func4 = self.layer_branch4[1](self.X_func4)
        else:
            y_func4 = self._net(
                self.X_func4, self.layer_branch4[1:], self.activation_branch4
            )
        # Branch net 5
        if callable(self.layer_branch5[1]):
            # User-defined network
            y_func5 = self.layer_branch5[1](self.X_func5)
        else:
            y_func5 = self._net(
                self.X_func5, self.layer_branch5[1:], self.activation_branch5
            )
        # Branch net 6
        if callable(self.layer_branch6[1]):
            # User-defined network
            y_func6 = self.layer_branch6[1](self.X_func6)
        else:
            y_func6 = self._net(
                self.X_func6, self.layer_branch6[1:], self.activation_branch6
            )
        # Branch net 7
        if callable(self.layer_branch7[1]):
            # User-defined network
            y_func7 = self.layer_branch7[1](self.X_func7)
        else:
            y_func7 = self._net(
                self.X_func7, self.layer_branch7[1:], self.activation_branch7
            )
        # Trunk net
        y_loc = self._net(self.X_loc, self.layer_trunk[1:], self.activation_trunk)

        # Dot product
        self.y = tf.multiply(y_func1, y_func2)
        self.y = tf.multiply(self.y, y_func3)
        self.y = tf.multiply(self.y, y_func4)
        self.y = tf.multiply(self.y, y_func5)
        self.y = tf.multiply(self.y, y_func6)
        self.y = tf.multiply(self.y, y_func7)
        self.y = tf.einsum("ip,jp->ij", self.y, y_loc)

        b = tf.Variable(tf.zeros(1))
        self.y += b
        self.target = tf.placeholder(config.real(tf), [None, None])
        self.built = True

