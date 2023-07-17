#!/bin/bash
for ((i=1; i<=16; i++ ))
do
echo -------- digits.npy --------
python3 over_jia.py Dataset/digits.npy Dataset/digits_attack.npy $i 11 100 10
echo ----------------------------
done

for ((i=1; i<=190; i+=2 ))
do
echo -------- letter.npy --------
python3 over_jia.py Dataset/letter.npy Dataset/letter_attack.npy $i 100 181 10
echo ----------------------------
done

for ((i=1; i<=97; i+=2 ))
do
echo -------- HAR.npy --------
python3 over_jia.py Dataset/HAR.npy Dataset/HAR_attack.npy $i 150 951 100
echo ----------------------------
done


for ((i=1; i<=179; i+=2 ))
do
echo -------- MNIST.npy --------
python3 over_jia.py Dataset/MNIST.npy Dataset/MNIST_attack.npy $i 1000 5001 500
echo ----------------------------
done


for ((i=1; i<=135; i+=2 ))
do
echo -------- cifar10.npy --------
python3 over_jia.py Dataset/cifar10.npy Dataset/cifar10_attack.npy $i 1000 5001 500
echo ----------------------------
done
