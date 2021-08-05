import nn

class PerceptronModel(object):
    def __init__(self, dimensions):
        """
        Initialize a new Perceptron instance.

        A perceptron classifies data points as either belonging to a particular
        class (+1) or not (-1). `dimensions` is the dimensionality of the data.
        For example, dimensions=2 would mean that the perceptron must classify
        2D points.
        """
        self.w = nn.Parameter(1, dimensions)

    def get_weights(self):
        """
        Return a Parameter instance with the current weights of the perceptron.
        """
        return self.w

    def run(self, x):
        """
        Calculates the score assigned by the perceptron to a data point x.

        Inputs:
            x: a node with shape (1 x dimensions)
        Returns: a node containing a single number (the score)
        """
        "*** YOUR CODE HERE ***"
        return nn.DotProduct(self.w,x)

    def get_prediction(self, x):
        """
        Calculates the predicted class for a single data point `x`.

        Returns: 1 or -1
        """
        "*** YOUR CODE HERE ***"
        z = nn.as_scalar(self.run(x))
        return 2*int(z>=0) - 1

    def train(self, dataset):
        """
        Train the perceptron until convergence.
        """
        "*** YOUR CODE HERE ***"
        converged = False
        while not converged:
            converged = True
            for x,y in dataset.iterate_once(1):
                z = self.get_prediction(x)
                if z != nn.as_scalar(y):
                    converged = False
                    self.w.update(x,nn.as_scalar(y))

class RegressionModel(object):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """
    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"

        # model architecture
        hidden_layer_dims = [(300,200),(200,100)]
        input_dim = [(1,hidden_layer_dims[0][0])]
        output_dim = [(hidden_layer_dims[-1][1],1)]
        layer_dims = input_dim + hidden_layer_dims + output_dim

        # instantiate layers as parameter objects
        self.layers = [nn.Parameter(dim[0],dim[1]) for dim in layer_dims]
        self.num_layers = len(self.layers)
        self.biases = [nn.Parameter(1,dim[1]) for dim in layer_dims]

        # training parameters
        self.loss_fn = nn.SquareLoss
        self.learning_rate = .05
        self.batch_size = 10
        self.max_epochs = 1000
        self.loss_threshold = 0.02/10

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
        Returns:
            A node with shape (batch_size x 1) containing predicted y-values
        """
        "*** YOUR CODE HERE ***"
        layer_input = x
        layer_output = None
        for i in range(self.num_layers):
            weights = self.layers[i]
            bias = self.biases[i]
            if i < self.num_layers - 1:
                layer_output = nn.ReLU(nn.AddBias(nn.Linear(layer_input, weights), bias))
                layer_input = layer_output
            else:
                layer_output = nn.AddBias(nn.Linear(layer_input, weights), bias)

        return layer_output


    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
            y: a node with shape (batch_size x 1), containing the true y-values
                to be used for training
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        pred_y = self.run(x)
        return self.loss_fn(pred_y, y)

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        loss = float('inf')
        epochs = 0
        while loss > self.loss_threshold and epochs < self.max_epochs:

            epochs += 1

            for x,y in dataset.iterate_once(self.batch_size):

                # Compute loss
                loss = self.get_loss(x, y)

                # Compute gradients
                gradients = nn.gradients(loss, self.layers+self.biases)
                layer_gradients = gradients[:self.num_layers]
                bias_gradients = gradients[self.num_layers:]

                # Update weights and biases
                for i in range(self.num_layers):
                    self.layers[i].update(layer_gradients[i], -1*self.learning_rate)
                    self.biases[i].update(bias_gradients[i], -1*self.learning_rate)

                # add loss and number of samples considered to running totals
                loss = nn.as_scalar(loss)

class DigitClassificationModel(object):
    """
    A model for handwritten digit classification using the MNIST dataset.

    Each handwritten digit is a 28x28 pixel grayscale image, which is flattened
    into a 784-dimensional vector for the purposes of this model. Each entry in
    the vector is a floating point number between 0 and 1.

    The goal is to sort each digit into one of 10 classes (number 0 through 9).

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"

        # model architecture
        hidden_layer_dims = [(300,200),(200,100)]
        input_dim = [(784,hidden_layer_dims[0][0])]
        output_dim = [(hidden_layer_dims[-1][1],10)]
        layer_dims = input_dim + hidden_layer_dims + output_dim

        # instantiate layers as parameter objects
        self.layers = [nn.Parameter(dim[0],dim[1]) for dim in layer_dims]
        self.num_layers = len(self.layers)
        self.biases = [nn.Parameter(1,dim[1]) for dim in layer_dims]

        # training parameters
        self.loss_fn = nn.SoftmaxLoss
        self.learning_rate = .05
        self.batch_size = 10
        self.max_epochs = 100
        self.target_accuracy = 0.975

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Your model should predict a node with shape (batch_size x 10),
        containing scores. Higher scores correspond to greater probability of
        the image belonging to a particular class.

        Inputs:
            x: a node with shape (batch_size x 784)
        Output:
            A node with shape (batch_size x 10) containing predicted scores
                (also called logits)
        """
        "*** YOUR CODE HERE ***"
        layer_input = x
        layer_output = None
        for i in range(self.num_layers):
            weights = self.layers[i]
            bias = self.biases[i]
            if i < self.num_layers - 1:
                layer_output = nn.ReLU(nn.AddBias(nn.Linear(layer_input, weights), bias))
                layer_input = layer_output
            else:
                layer_output = nn.AddBias(nn.Linear(layer_input, weights), bias)

        return layer_output

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 10). Each row is a one-hot vector encoding the correct
        digit class (0-9).

        Inputs:
            x: a node with shape (batch_size x 784)
            y: a node with shape (batch_size x 10)
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        y_pred = self.run(x)
        return self.loss_fn(y_pred,y)

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        accuracy = 0
        epochs = 0
        while accuracy < self.target_accuracy and epochs < self.max_epochs:

            epochs += 1

            for x,y in dataset.iterate_once(self.batch_size):

                # Compute loss
                loss = self.get_loss(x, y)

                # Compute gradients
                gradients = nn.gradients(loss, self.layers+self.biases)
                layer_gradients = gradients[:self.num_layers]
                bias_gradients = gradients[self.num_layers:]

                # Update weights and biases
                for i in range(self.num_layers):
                    self.layers[i].update(layer_gradients[i], -1*self.learning_rate)
                    self.biases[i].update(bias_gradients[i], -1*self.learning_rate)

            # evaluate model accuracy on validation set
            accuracy = dataset.get_validation_accuracy()


class DeepQModel(object):
    """
    A model that uses a Deep Q-value Network (DQN) to approximate Q(s,a) as part
    of reinforcement learning.
    """
    def __init__(self, state_dim, action_dim):
        self.num_actions = action_dim
        self.state_size = state_dim

        # Remember to set self.learning_rate, self.numTrainingGames,
        # self.parameters, and self.batch_size!
        "*** YOUR CODE HERE ***"

        # model architecture
        input_layer = [(self.state_size, 200),(1, 200)]  # input layer weight and bias dimensions
        hidden_layers = [(200, 200),(1,200),(200,200),(1,200)]   # hidden layer weight and bias dimensions
        output_layer = [(200,self.num_actions),(1,self.num_actions)]  # output layer weight and bias dimensions
        layer_dims = input_layer + hidden_layers + output_layer
        self.parameters = [nn.Parameter(dim[0],dim[1]) for dim in layer_dims]

        # training parameters
        self.learning_rate = 0.05
        self.numTrainingGames = 10000
        self.batch_size = 100

    def get_loss(self, states, Q_target):
        """
        Returns the Squared Loss between Q values currently predicted 
        by the network, and Q_target.
        Inputs:
            states: a node with shape (batch_size x state_dim)
            Q_target: a (batch_size x num_actions) numpy array, or None
        Output:
            loss node between Q predictions and Q_target
        """
        "*** YOUR CODE HERE ***"
        Q_pred = self.run(states)
        return nn.SquareLoss(Q_pred,Q_target)

    def run(self, states):
        """
        Runs the DQN for a batch of states.
        The DQN takes the state and returns the Q-values for all possible actions
        that can be taken. That is, if there are two actions, the network takes
        as input the state s and computes the vector [Q(s, a_1), Q(s, a_2)]
        Inputs:
            states: a node with shape (batch_size x state_dim)
        Output:
            result: a node with shape (batch_size x num_actions) containing Q-value
                scores for each of the actions
        """
        "*** YOUR CODE HERE ***"
        layer_input = states
        layer_output = None
        for i in range(len(self.parameters)-2):
            term = self.parameters[i]
            # weight node
            if i % 2 == 0:
                layer_output = nn.Linear(layer_input,term)
            # bias node
            else:
                layer_output = nn.ReLU(nn.AddBias(layer_output,term))
                layer_input = layer_output

        layer_output = nn.AddBias(nn.Linear(layer_output,self.parameters[-2]),self.parameters[-1])
        return layer_output

    def gradient_update(self, states, Q_target):
        """
        Update your parameters by one gradient step with the .update(...) function.
        Inputs:
            states: a node with shape (batch_size x state_dim)
            Q_target: a (batch_size x num_actions) numpy array, or None
        Output:
            None
        """
        "*** YOUR CODE HERE ***"
        loss = self.get_loss(states,Q_target)
        gradients = nn.gradients(loss, self.parameters)
        for i in range(len(self.parameters)):
            grad = gradients[i]
            self.parameters[i].update(grad, -1*self.learning_rate)