# Preliminary prototype work with activation relaxation on MNIST.

import argparse
import subprocess
from copy import deepcopy
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms


def get_dataset(batch_size, norm_factor, dataset="mnist"):
    # currently assuming just MNIST
    transform = transforms.Compose([transforms.ToTensor()])  # , transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])

    if dataset == "mnist":
        trainset = torchvision.datasets.MNIST(root='data', download=True, transform=transform)
        print("trainset: ", trainset)

        trainset = list(iter(torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)))

        testset = torchvision.datasets.MNIST(root='data', train=False, download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True)
        testset = list(iter(testloader))
        for i, (img, label) in enumerate(trainset):
            trainset[i] = (img.reshape(len(img), 784) / norm_factor, label)
        for i, (img, label) in enumerate(testset):
            testset[i] = (img.reshape(len(img), 784) / norm_factor, label)
        return trainset, testset
    elif dataset == "svhn":
        trainset = torchvision.datasets.SVHN(root='./svhn_data', transform=transform)
        print("trainset: ", trainset)

        trainset = list(iter(torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)))

        testset = torchvision.datasets.SVHN(root='./svhn_data', split='test', transform=transform)
        testset = list(iter(torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True)))
        for i, (img, label) in enumerate(trainset):
            trainset[i] = (img.reshape(len(img), 3 * 32 * 32) / norm_factor, label)
        for i, (img, label) in enumerate(testset):
            testset[i] = (img.reshape(len(img), 3 * 32 * 32) / norm_factor, label)
        return trainset, testset
    elif dataset == "fashion":
        trainset = torchvision.datasets.FashionMNIST(root='./fashion_data', transform=transform)

        print("trainset: ", trainset)
        trainset = list(iter(torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=False)))

        testset = torchvision.datasets.FashionMNIST(root='./fashion_data', train=False,download=True, transform=transform)
        testset = list(iter(torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True)))
        for i, (img, label) in enumerate(trainset):
            trainset[i] = (img.reshape(len(img), 784) / norm_factor, label)
        for i, (img, label) in enumerate(testset):
            testset[i] = (img.reshape(len(img), 784) / norm_factor, label)
        return trainset, testset
    else:
        raise ValueError("Dataset not recognised -- must be mnist, svhn, or fashion")


def np_norm(x):
    return np.linalg.norm(x.numpy())

def onehot(x):
    l: int = len(x)
    z = torch.zeros([l, 10])
    for i in range(l): z[i, x[i]] = 1
    return z.float().to(DEVICE)

### functions ###
def set_tensor(xs):    return xs.float().to(DEVICE)


def tanh(xs): return torch.tanh(xs)
def tanh_deriv(xs): return 1.0 - torch.tanh(xs) ** 2.0

def linear(x):    return x
def linear_deriv(x):
    return set_tensor(torch.ones_like(x))
    # TODO may be faster if allowed to modify x
    # x.fill(1)
    # set_tensor(x)


def relu(xs): return torch.clamp(xs, min=0)

def relu_deriv(xs):
    rel = relu(xs)
    rel[rel > 0] = 1
    return rel


def softmax(xs): return F.softmax(xs)

def sigmoid(xs): return F.sigmoid(xs)

def sigmoid_deriv(x):
    sx = F.sigmoid(x)
    return sx * (torch.ones_like(x) - sx)

def mse_loss(out, label): return torch.sum((out - label) ** 2)
def mse_deriv(out, label): return 2 * (out - label)

ce_loss = nn.CrossEntropyLoss()
def cross_entropy_loss(out, label): return ce_loss(out, label)
def cross_entropy_deriv(out, label): return out - label

def my_cross_entropy(out, label): return -torch.sum(label * torch.log(out + 1e-6))

def loss(loss_arg):
    if loss_arg == "mse":
        return mse_loss, mse_deriv
    elif loss_arg == "crossentropy":
        return my_cross_entropy, cross_entropy_deriv
    else:
        raise ValueError(
            "loss argument not expected. Can be one of 'mse' and 'crossentropy'. You inputted " + str(loss_arg))


def boolcheck(x):
    return str(x).lower() in ["true", "1", "yes"]


class FCLayer(object):
    def __init__(self, input_size, output_size, batch_size, fn, fn_deriv, inference_lr, weight_lr, device='cpu',
                 numerical_test=False):
        self.input_size = input_size
        self.output_size = output_size
        self.batch_size = batch_size
        self.fn = fn
        self.fn_deriv = fn_deriv
        self.inference_lr = inference_lr
        self.weight_lr = weight_lr
        self.device = device
        self.weights = torch.empty((input_size, output_size)).normal_(0.0, 0.05).float().to(self.device)
        self.numerical_test = numerical_test
        self.bias = torch.zeros((batch_size, output_size)).float().to(self.device)
        self.x = self.x0 = None
        self.act = None
        self.dAct = None
        self.backwards_weights = None
        self.y = None
        if self.numerical_test:
            self.weights = nn.Parameter(self.weights)
            self.bias = nn.Parameter(self.bias)

    def forward(self, x: torch.Tensor):
        self.x = x.clone()
        self.x0 = x   #self.x.clone()  #clone necessary?
        self.act = x @ self.weights
        self.y = self.fn(self.act + self.bias)
        return self.y

    def set_weight_parameter(self):
        self.weights = nn.Parameter(self.weights)
        self.bias = nn.Parameter(self.bias)

    def detach_weight_parameter(self):
        self.weights = self.weights.detach()
        self.bias = self.bias.detach()

    def init_backwards_weights(self):
        self.backwards_weights = torch.empty((self.output_size, self.input_size)).normal_(0.0, 0.05).float().to(self.device)

    def backward(self, xx):

        self.dAct = self.fn_deriv(self.act)

        if self.use_backward_nonlinearity:
            y = xx * self.dAct
        else:
            y = xx

        W = self.backwards_weights if self.use_backwards_weights else self.weights.T

        x = self.x
        x -= self.inference_lr * (x - (y @ W))

        return x

    def activationDeriv(self):
        return self.fn_deriv(self.act)

    def update_weights(self, xx, update_weights=True):
        actDeriv = self.dAct
        self.dAct = None #acquire

        wgrad = self.x0.T @ (xx * actDeriv)
        biasgrad = xx * actDeriv

        if self.use_backwards_weights and self.update_backwards_weights:
            bwgrad = wgrad.clone().T
        else:
            bwgrad = None

        if self.numerical_test:
            print("Wgrad: ", wgrad.shape)
            print(wgrad[0:10, 0])
            print("bias grad: ", biasgrad.shape)
            print(biasgrad[0:10, 0])

        if update_weights:
            wlr = self.weight_lr
            self.weights -= wlr * wgrad
            self.bias    -= wlr * biasgrad

            if self.use_backwards_weights and self.update_backwards_weights:
                self.backwards_weights -= wlr * bwgrad

        return wgrad


class Net(object):
    layers: list

    def __init__(self, layers, n_inference_steps, use_backwards_weights, update_backwards_weights,
                 use_backward_nonlinearity, store_gradient_angle=False, loss_fn=None, loss_fn_deriv=None, device="cpu"):
        self.layers = layers
        self.n_inference_steps = n_inference_steps
        self.use_backwards_weights = use_backwards_weights
        self.update_backwards_weights = update_backwards_weights
        self.use_backward_nonlinearity = use_backward_nonlinearity
        self.store_gradient_angle = store_gradient_angle
        self.loss_fn = loss_fn
        self.loss_fn_deriv = loss_fn_deriv
        self.device = device
        self.update_layer_params()
        # check that the correct things are getting called
        # subprocess.call(["echo", "Params checking: " + str(self.use_backwards_weights) + " " + str(
        #    self.use_backward_nonlinearity) + " " + str(self.update_backwards_weights)])

    def forward(self, inp):
        L = len(self.layers)
        xs = [[] for _ in range(L + 1)]
        xs[0] = inp
        for i, l in enumerate(self.layers):
            xs[i + 1] = l.forward(xs[i])
        return xs[L]

    def set_net_params(self):
        for l in self.layers: l.set_weight_parameter()

    def detach_net_params(self):
        for l in self.layers: l.detach_weight_parameter()

    def unset_numerical_test(self):
        for l in self.layers: l.numerical_test = False

    def update_layer_params(self):
        for l in self.layers:
            l.use_backwards_weights = self.use_backwards_weights
            l.update_backwards_weights = self.update_backwards_weights
            l.use_backward_nonlinearity = self.use_backward_nonlinearity
            if self.use_backwards_weights:
                l.init_backwards_weights()

    def numerical_check(self, inp, label):
        for l in self.layers:
            l.set_weight_parameter()
            l.numerical_test = True
        out = self.forward(inp)
        L = torch.sum((out - label) ** 2)
        L.backward()
        print("Backprop gradients")
        for l in self.layers:
            print("BP weight grad: ", l.weights.grad.shape)
            print(l.weights.grad[0:10, 0])
            print("BP bias grad: ", l.bias.grad.shape)
            print(l.bias.grad[0:10, 0])
        print("AR gradients")
        self.put(inp, label, self.n_inference_steps, update_weights=False)

    """ train, learn """

    def put(self, inps, labels, num_inference_steps, update_weights=True):
        # print("learn batch update weights: ", update_weights)
        L = len(self.layers)

        # forward pass
        fwd = [[] for _ in range(L + 1)]
        fwd[0] = inps
        for i, l in enumerate(self.layers):
            fwd[i + 1] = l.forward(fwd[i])

        err = self.loss_fn_deriv(fwd[L], labels)

        rev = [[] for _ in range(L + 1)]
        rev[L] = err
        for n in range(num_inference_steps):
            # backward inference
            for j in reversed(range(L)):
                rev[j] = self.layers[j].backward(rev[j + 1])

        # weight updates
        wgrads = []
        for i, l in enumerate(self.layers):
            wgrad = l.update_weights(rev[i + 1], update_weights=update_weights)
            if self.store_gradient_angle:
                wgrads.append(wgrad)

        return wgrads

    @staticmethod
    def save_model(logdir, savedir, losses, accs, test_accs, grad_angles, mean_test_accs, mean_train_accs):
        np.save(logdir + "/losses.npy", np.array(losses))
        np.save(logdir + "/accs.npy", np.array(accs))
        np.save(logdir + "/test_accs.npy", np.array(test_accs))
        np.save(logdir + "/grad_angles.npy", np.array(grad_angles))
        np.save(logdir + "/mean_test_accs.npy", np.array(mean_test_accs))
        np.save(logdir + "/mean_train_accs.npy", np.array(mean_train_accs))
        subprocess.call(['rsync', '--archive', '--update', '--compress', '--progress', str(logdir) + "/", str(savedir)])
        print("Rsynced files from: " + str(logdir) + "/ " + " to" + str(savedir))
        now = datetime.now()
        current_time = str(now.strftime("%H:%M:%S"))
        subprocess.call(['echo', 'saved at time: ' + str(current_time)])

    def compute_gradient_angle(self, img, label):
        # img = img.to(self.device)
        # label = onehot(label).to(self.device)
        self.detach_net_params()
        AR_grads = self.put(img, label, self.n_inference_steps, update_weights=False)
        # compute BP gradients
        self.set_net_params()
        img = nn.Parameter(img)
        out = self.forward(img)
        if self.loss_fn == cross_entropy_loss:
            L = self.loss_fn(out, label)
        else:
            L = torch.sum((out - label) ** 2)
        L.backward()
        BP_grads = [deepcopy(l.weights.grad) for l in self.layers]
        self.detach_net_params()
        # accumulate mean angle
        mean_angle = 0
        for (AR_grad, BP_grad) in zip(AR_grads, BP_grads):
            # lillicrap approach of simply computing the cosine similarity on the flattened gradient matrices
            AR_flat = AR_grad.detach().flatten().cpu()
            BP_flat = BP_grad.detach().flatten().cpu()
            angle = (torch.dot(AR_flat, BP_flat) / (np_norm(AR_flat) * np_norm(BP_flat))).item()
            angle = np.arccos(min(angle, 1))  # hack for stability
            mean_angle += angle
        return mean_angle / len(self.layers)

    def train(self, trainset, epochs):
        return self.trainAR(epochs, trainset)

    def trainAR(self, epochs, trainset):
        self.unset_numerical_test()
        losses = []
        # accs = []
        # test_accs = []
        gradient_angles = []
        # mean_test_accs = []
        # mean_train_accs = []
        # begin training loop
        #device = self.device
        for n_epoch in range(epochs):
            print("epoch ", n_epoch)
            for n, (x, y) in enumerate(trainset):
                #with torch.no_grad():

                if self.loss_fn != cross_entropy_loss:
                    y = onehot(y)
                else:
                    y = y.long()

                #x = x.to(device)
                #y = y.to(device)

                self.put(x, y, self.n_inference_steps)

                z = self.forward(x)

                loss: float = self.loss_fn(y, z).item()
                lossPerPx = loss / len(x)

                print("\tloss/px %f" % lossPerPx)
                losses.append(loss)
                # accs.append(acc)
                # epoch_train_accs.append(accs)

                if self.store_gradient_angle:
                    gradient_angles.append(self.compute_gradient_angle(x, y))

            # if test:
            #     #mean_train_accs.append(np.mean(np.array(epoch_train_accs)))
            #     epoch_test_accs = []
            #     for tn, (test_img, test_label) in enumerate(testset):
            #         test_img = test_img.to(self.device)
            #         labels = onehot(test_label).to(self.device)
            #         pred_outs = self.forward(test_img)
            #         #test_acc = accuracy(pred_outs, labels)
            #         #test_accs.append(test_acc)
            #         #epoch_test_accs.append(test_acc)
            #     #mean_test_accs.append(np.mean(np.array(epoch_test_accs)))
            #     #self.save_model(logdir, savedir, losses, accs, test_accs, gradient_angles, mean_test_accs, mean_train_accs)
        return losses


class BackpropNet(object):
    def __init__(self, layers, loss_fn=None, device="cpu"):
        self.layers = layers
        self.device = device
        self.loss_fn = loss_fn
        for l in self.layers:
            l.set_weight_parameter()
            l.numerical_test = True

    def forward(self, x):
        for i, l in enumerate(self.layers):
            x = l.forward(x)
        return x

    @staticmethod
    def save_model(logdir, savedir, losses, accs, test_accs, mean_test_accs, mean_train_accs):
        np.save(logdir + "/losses.npy", np.array(losses))
        np.save(logdir + "/accs.npy", np.array(accs))
        np.save(logdir + "/test_accs.npy", np.array(test_accs))
        np.save(logdir + "/mean_test_accs.npy", np.array(mean_test_accs))
        np.save(logdir + "/mean_train_accs.npy", np.array(mean_train_accs))
        subprocess.call(['rsync', '--archive', '--update', '--compress', '--progress', str(logdir) + "/", str(savedir)])
        print("Rsynced files from: " + str(logdir) + "/ " + " to" + str(savedir))
        now = datetime.now()
        current_time = str(now.strftime("%H:%M:%S"))
        subprocess.call(['echo', 'saved at time: ' + str(current_time)])

    def zero_grad(self):
        for l in self.layers:
            l.weights.grad.fill(0)

    def train(self, trainset, num_epochs):
        return self.trainBP(num_epochs, trainset)

    def trainBP(self, num_epochs, trainset):
        losses = []
        # accs = []
        # test_accs = []
        # mean_train_accs = []
        # mean_test_accs = []
        # begin training loop
        lossFn = self.loss_fn
        device = self.device
        for n_epoch in range(num_epochs):
            # epoch_train_accs = []
            # print("Beginning epoch ",n_epoch)
            # acclist = []
            losslist = []
            # mean_train_accs.append(np.mean(np.array(epoch_train_accs)))
            for n, (x, y) in enumerate(trainset):
                x = x.to(device)

                if lossFn != cross_entropy_loss:
                    y = onehot(y).to(device)
                else:
                    y = y.long().to(device)

                z = self.forward(x)

                if lossFn:
                    L = lossFn(y, z)
                else:
                    L = torch.sum((y - z) ** 2)

                L.backward()

                loss = L.item()

                # SGD update
                for l in self.layers:
                    weights = l.weights
                    l.weights = nn.Parameter(
                        weights.detach().clone() - l.weight_lr * weights.grad.clone()
                    )

                # zero grad weights just to be sure
                # self.zero_grad()

                # acc = accuracy(y, z)
                print("%d\tloss/px %f" % (n_epoch, loss / len(z)))
                losslist.append(loss)
                # acclist.append(acc)
                # acclist.append(-L)
                # epoch_train_accs.append(accs)

            losses.append(np.mean(np.array(losslist)))

            # if test:
            #     with torch.no_grad():
            #         epoch_test_accs = []
            #         for tn, (test_img, test_label) in enumerate(testset):
            #             test_img = test_img.to(device)
            #             labels = onehot(test_label).to(device)
            #             z = self.forward(test_img)
            #             test_acc = accuracy(z, labels)
            #             #test_accs.append(test_acc)
            #             epoch_test_accs.append(test_acc)
            # accs.append(np.mean(np.array(acclist)))
            # if save:
            #    mean_test_accs.append(np.mean(np.array(epoch_test_accs)))
            #    self.save_model(logdir, savedir, losses, accs, test_accs, mean_test_accs, mean_train_accs)
        return losses


if __name__ == '__main__':
    global DEVICE
    DEVICE: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    parser = argparse.ArgumentParser()
    parser.add_argument("--network_type", type=str, default="ar")
    parser.add_argument("--loss_fn", type=str, default="mse")

    parser.add_argument("--learning_rate", type=float, default=0.01)
    parser.add_argument("--inference_learning_rate", type=float, default=1)
    parser.add_argument("--n_inference_steps", type=int, default=1)

    parser.add_argument("--batch_size", type=int, default=16)

    parser.add_argument("--norm_factor", type=float, default=1)

    parser.add_argument("--dataset", type=str, default="mnist")
    parser.add_argument("--N_epochs", type=int, default=50)

    parser.add_argument("--use_backwards_weights", type=boolcheck, default=False)
    parser.add_argument("--update_backwards_weights", type=boolcheck, default=False)
    parser.add_argument("--use_backward_nonlinearity", type=boolcheck, default=False)
    parser.add_argument("--store_gradient_angle", type=boolcheck, default=False)

    parser.add_argument("--logdir", type=str, default="logs")
    parser.add_argument("--savedir", type=str, default="savedir")

    args = parser.parse_args()
    print(args)

    # create folders
    if args.savedir != "":  subprocess.call(["mkdir", "-p", str(args.savedir)])
    if args.logdir  != "":  subprocess.call(["mkdir", "-p", str(args.logdir)])

    loss_fn, loss_fn_deriv = loss(args.loss_fn)

    trainset, testset = get_dataset(args.batch_size, args.norm_factor, dataset=args.dataset)

    learnRate, infRate = args.learning_rate, args.inference_learning_rate

    if args.dataset == "mnist" or args.dataset == "fashion":
        l1 = FCLayer(784, 300, args.batch_size, relu, relu_deriv, infRate, learnRate, device=DEVICE)
        l2 = FCLayer(300, 300, args.batch_size, relu, relu_deriv, infRate, learnRate, device=DEVICE)
        l3 = FCLayer(300, 100, args.batch_size, relu, relu_deriv, infRate, learnRate, device=DEVICE)
        if args.loss_fn == "crossentropy":
            # softmax_deriv?
            l4 = FCLayer(100, 10, args.batch_size, softmax, linear_deriv, infRate,learnRate, device=DEVICE)
        else:
            l4 = FCLayer(100, 10, args.batch_size, linear, linear_deriv, infRate, learnRate, device=DEVICE)
    elif args.dataset == "svhn":
        l1 = FCLayer(3072, 1000, args.batch_size, relu, relu_deriv, infRate, learnRate, device=DEVICE)
        l2 = FCLayer(1000, 1000, args.batch_size, relu, relu_deriv, infRate, learnRate, device=DEVICE)
        l3 = FCLayer(1000, 300, args.batch_size, relu, relu_deriv, infRate, learnRate, device=DEVICE)
        if args.loss_fn == "crossentropy":
            # softmax_deriv?
            l4 = FCLayer(100, 10, args.batch_size, softmax, linear_deriv, infRate,learnRate, device=DEVICE)
        else:
            l4 = FCLayer(100, 10, args.batch_size, linear, linear_deriv, infRate,learnRate, device=DEVICE)
    else:
        raise ValueError("dataset not recognised")

    layers = [l1, l2, l3, l4]

    print(layers)

    if args.network_type == "ar":
        net = Net(layers, args.n_inference_steps, use_backwards_weights=args.use_backwards_weights,
                  update_backwards_weights=args.update_backwards_weights,
                  use_backward_nonlinearity=args.use_backward_nonlinearity,
                  store_gradient_angle=args.store_gradient_angle, loss_fn=loss_fn, loss_fn_deriv=loss_fn_deriv,
                  device=DEVICE)
    elif args.network_type == "bp":
        net = BackpropNet(layers, loss_fn=loss_fn, device=DEVICE)
    else:
        raise ValueError("Network type not recognised: must be either ar (activation relaxation) or bp (backprop)")

    net.train(trainset[0:-2], 2)
