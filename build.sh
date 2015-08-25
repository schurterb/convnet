
echo "Performing convnet build routine..."

echo "Building MALIS"
python malis/setup.py build_ext --inplace

echo "Initializing test, train, predict scripts"
cp train.py train
chmod 755 train

cp predict.py predict
chmod 755 predict

cp test.py test
chmod 755 test

echo "Build complete."