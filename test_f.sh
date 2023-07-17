#!/bin/bash
for i in {1..10}
do
echo -------- iris.npy --------
python3 falsify_all_test_refine.py Dataset/iris.npy Dataset/iris_attack.npy $i 1 50 6 1800
echo -------- digits.npy --------
python3 falsify_all_test_refine.py Dataset/digits.npy Dataset/digits_attack.npy $i 11 100 10 1800
echo -------- letter.npy --------
python3 falsify_all_test_refine.py Dataset/letter.npy Dataset/letter_attack.npy $i 100 181 10 1800
echo -------- HAR.npy --------
python3 falsify_all_test_refine.py Dataset/HAR.npy Dataset/HAR_attack.npy $i 150 951 100 1800
echo -------- MNIST.npy --------
python3 falsify_all_test_refine.py Dataset/MNIST.npy Dataset/MNIST_attack.npy $i 1000 5001 500 1800
echo -------- cifar10.npy --------
python3 falsify_all_test_refine.py Dataset/cifar10.npy Dataset/cifar10_attack.npy $i 1000 5001 500  1800
echo ----------------------------
done
