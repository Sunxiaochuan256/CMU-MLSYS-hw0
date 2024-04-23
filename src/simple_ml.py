import struct
import numpy as np
import gzip
try:
    from simple_ml_ext import *
except:
    pass


def add(x, y):
    """ A trivial 'add' function you should implement to get used to the
    autograder and submission system.  The solution to this problem is in the
    the homework notebook.

    Args:
        x (Python number or numpy array)
        y (Python number or numpy array)

    Return:
        Sum of x + y
    """
    ### BEGIN YOUR CODE
    return x - y
    ### END YOUR CODE


def parse_mnist(image_filename, label_filename):
    """ Read an images and labels file in MNIST format.  See this page:
    http://yann.lecun.com/exdb/mnist/ for a description of the file format.

    Args:
        image_filename (str): name of gzipped images file in MNIST format
        label_filename (str): name of gzipped labels file in MNIST format

    Returns:
        Tuple (X,y):
            X (numpy.ndarray[np.float32]): 2D numpy array containing the loaded 
                data.  The dimensionality of the data should be 
                (num_examples x input_dim) where 'input_dim' is the full 
                dimension of the data, e.g., since MNIST images are 28x28, it 
                will be 784.  Values should be of type np.float32, and the data 
                should be normalized to have a minimum value of 0.0 and a 
                maximum value of 1.0 (i.e., scale original values of 0 to 0.0 
                and 255 to 1.0).

            y (numpy.ndarray[dtype=np.uint8]): 1D numpy array containing the
                labels of the examples.  Values should be of type np.uint8 and
                for MNIST will contain the values 0-9.
    """
    ### BEGIN YOUR CODE
    # 打开并解压缩图像文件
    with gzip.open(image_filename, 'rb') as image_file:
        # 读取文件头，跳过前16字节
        image_file.read(16)
        # 读取图像数据
        image_data = image_file.read()

    # 打开并解压缩标签文件
    with gzip.open(label_filename, 'rb') as label_file:
        # 读取文件头，跳过前8字节
        label_file.read(8)
        # 读取标签数据
        label_data = label_file.read()

    # 将图像和标签数据转换为NumPy数组
    image_array = np.frombuffer(image_data, dtype=np.uint8)
    label_array = np.frombuffer(label_data, dtype=np.uint8)

    # 获取图像的维度
    num_images = len(label_array)
    image_dim = len(image_array) // num_images

    # 将图像数据转换为浮点数并归一化
    image_array = image_array.astype(np.float32) / 255.0

    # 返回图像数据和标签数据的元组
    return image_array.reshape(num_images, image_dim), label_array
    ### END YOUR CODE


def softmax_loss(Z, y):
    """ Return softmax loss.  Note that for the purposes of this assignment,
    you don't need to worry about "nicely" scaling the numerical properties
    of the log-sum-exp computation, but can just compute this directly.

    Args:
        Z (np.ndarray[np.float32]): 2D numpy array of shape
            (batch_size, num_classes), containing the logit predictions for
            each class.
        y (np.ndarray[np.uint8]): 1D numpy array of shape (batch_size, )
            containing the true label of each example.

    Returns:
        Average softmax loss over the sample.
    """
    ### BEGIN YOUR CODE
    # 确保输入的logits Z是浮点数，以便进行exp计算
    Z = Z.astype(np.float64)

    # 计算log-sum-exp，即对每个样本的logits应用exp，然后按行求和，最后取对数。
    # 这一步会得到每个样本的log-sum-exp值。
    log_sum_exp = np.log(np.sum(np.exp(Z), axis=1))

    # 从log-sum-exp中减去每个样本真实类别的logit。
    # np.arange(y.shape[0])生成一个索引数组，用于选择每个样本的真实类别的logit。
    correct_logit = Z[np.arange(y.shape[0]), y]

    # 计算每个样本的softmax损失
    softmax_loss_per_sample = log_sum_exp - correct_logit

    # 计算所有样本的平均softmax损失
    average_loss = np.mean(softmax_loss_per_sample)

    return average_loss
    ### END YOUR CODE


def softmax_regression_epoch(X, y, theta, lr = 0.1, batch=100):
    """ Run a single epoch of SGD for softmax regression on the data, using
    the step size lr and specified batch size.  This function should modify the
    theta matrix in place, and you should iterate through batches in X _without_
    randomizing the order.

    Args:
        X (np.ndarray[np.float32]): 2D input array of size
            (num_examples x input_dim).
        y (np.ndarray[np.uint8]): 1D class label array of size (num_examples,)
        theta (np.ndarrray[np.float32]): 2D array of softmax regression
            parameters, of shape (input_dim, num_classes)
        lr (float): step size (learning rate) for SGD
        batch (int): size of SGD minibatch

    Returns:
        None
    """
    ### BEGIN YOUR CODE
    num_examples = X.shape[0]
    i = 0
    
    while i + batch <= num_examples:
        inputs = X[i:i+batch]
        labels = y[i:i+batch]
        softmax_regression_batch(inputs, labels, theta, lr)
        i += batch
    
    if i < num_examples:
        inputs = X[i:num_examples]
        labels = y[i:num_examples]
        softmax_regression_batch(inputs, labels, theta, lr)
    ### END YOUR CODE

def softmax_regression_batch(X, y, theta, lr = 0.1):
    input_dim = X.shape[1]
    output_dim = theta.shape[1]
    one_hot = np.zeros((X.shape[0], theta.shape[1]), dtype=np.int8)
    np.put_along_axis(one_hot, np.expand_dims(y, axis=1), 1, axis=1)

    h = X@theta
    eh = np.exp(h)
    z = eh / eh.sum(axis=1, keepdims=True)
    gradient = X.transpose()@(z - one_hot) # (batch, output_dim)
    sub = gradient*lr / (X.shape[0])
    np.subtract(theta, sub, out=theta)

def nn_epoch(X, y, W1, W2, lr = 0.1, batch=100):
    """ Run a single epoch of SGD for a two-layer neural network defined by the
    weights W1 and W2 (with no bias terms):
        logits = ReLU(X * W1) * W2
    The function should use the step size lr, and the specified batch size (and
    again, without randomizing the order of X).  It should modify the
    W1 and W2 matrices in place.

    Args:
        X (np.ndarray[np.float32]): 2D input array of size
            (num_examples x input_dim).
        y (np.ndarray[np.uint8]): 1D class label array of size (num_examples,)
        W1 (np.ndarray[np.float32]): 2D array of first layer weights, of shape
            (input_dim, hidden_dim)
        W2 (np.ndarray[np.float32]): 2D array of second layer weights, of shape
            (hidden_dim, num_classes)
        lr (float): step size (learning rate) for SGD
        batch (int): size of SGD minibatch

    Returns:
        None
    """
    ### BEGIN YOUR CODE
    num_examples = X.shape[0]
    i = 0
    
    while i + batch <= num_examples:
        inputs = X[i:i+batch]
        labels = y[i:i+batch]
        nn_epoch_batch(inputs, labels, W1, W2, lr)
        i += batch
    
    if i < num_examples:
        inputs = X[i:num_examples]
        labels = y[i:num_examples]
        nn_epoch_batch(inputs, labels, W1, W2, lr)
    ### END YOUR CODE
		
def nn_epoch_batch(X, y, W1, W2, lr = 0.1, batch=100):
    Z1 = np.maximum(X@W1,0) # [batch, d]
    relu_mask = Z1 > 0
    h = Z1@W2 # [batch, k]
    eh = np.exp(h)
    norm_eh = eh / eh.sum(axis=1, keepdims=True) # [batch, k]    
    Iy_onehot = np.zeros((X.shape[0], W2.shape[1]), dtype=np.uint8)
    np.put_along_axis(Iy_onehot, np.expand_dims(y, axis=1), 1, axis=1)
    G2 = norm_eh - Iy_onehot # [batch, k]
    G1 = relu_mask * (G2@W2.transpose()) # [batch, d]

    gradient1 = X.transpose()@G1 / (X.shape[0])
    gradient2 = Z1.transpose()@G2 / (X.shape[0])
    
    np.subtract(W1, gradient1*lr, out=W1)
    np.subtract(W2, gradient2*lr, out=W2)



### CODE BELOW IS FOR ILLUSTRATION, YOU DO NOT NEED TO EDIT

def loss_err(h,y):
    """ Helper funciton to compute both loss and error"""
    return softmax_loss(h,y), np.mean(h.argmax(axis=1) != y)


def train_softmax(X_tr, y_tr, X_te, y_te, epochs=10, lr=0.5, batch=100,
                  cpp=False):
    """ Example function to fully train a softmax regression classifier """
    theta = np.zeros((X_tr.shape[1], y_tr.max()+1), dtype=np.float32)
    print("| Epoch | Train Loss | Train Err | Test Loss | Test Err |")
    for epoch in range(epochs):
        if not cpp:
            softmax_regression_epoch(X_tr, y_tr, theta, lr=lr, batch=batch)
        else:
            softmax_regression_epoch_cpp(X_tr, y_tr, theta, lr=lr, batch=batch)
        train_loss, train_err = loss_err(X_tr @ theta, y_tr)
        test_loss, test_err = loss_err(X_te @ theta, y_te)
        print("|  {:>4} |    {:.5f} |   {:.5f} |   {:.5f} |  {:.5f} |"\
              .format(epoch, train_loss, train_err, test_loss, test_err))


def train_nn(X_tr, y_tr, X_te, y_te, hidden_dim = 500,
             epochs=10, lr=0.5, batch=100):
    """ Example function to train two layer neural network """
    n, k = X_tr.shape[1], y_tr.max() + 1
    np.random.seed(0)
    W1 = np.random.randn(n, hidden_dim).astype(np.float32) / np.sqrt(hidden_dim)
    W2 = np.random.randn(hidden_dim, k).astype(np.float32) / np.sqrt(k)

    print("| Epoch | Train Loss | Train Err | Test Loss | Test Err |")
    for epoch in range(epochs):
        nn_epoch(X_tr, y_tr, W1, W2, lr=lr, batch=batch)
        train_loss, train_err = loss_err(np.maximum(X_tr@W1,0)@W2, y_tr)
        test_loss, test_err = loss_err(np.maximum(X_te@W1,0)@W2, y_te)
        print("|  {:>4} |    {:.5f} |   {:.5f} |   {:.5f} |  {:.5f} |"\
              .format(epoch, train_loss, train_err, test_loss, test_err))



if __name__ == "__main__":
    X_tr, y_tr = parse_mnist("data/train-images-idx3-ubyte.gz",
                             "data/train-labels-idx1-ubyte.gz")
    X_te, y_te = parse_mnist("data/t10k-images-idx3-ubyte.gz",
                             "data/t10k-labels-idx1-ubyte.gz")

    print("Training softmax regression")
    train_softmax(X_tr, y_tr, X_te, y_te, epochs=10, lr = 0.1)

    print("\nTraining two layer neural network w/ 100 hidden units")
    train_nn(X_tr, y_tr, X_te, y_te, hidden_dim=100, epochs=20, lr = 0.2)
