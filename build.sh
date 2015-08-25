
echo "Performing convnet build routine..."

echo "Building MALIS"
python malis/setup.py build_ext --inplace

echo "Initializing test, train, predict scripts"
cp train.py train

cp predict.py predict

cp test.py test

echo "Build complete."