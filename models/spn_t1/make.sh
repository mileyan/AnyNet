rm -rf build
rm -rf gate_lib
python setup.py clean
python setup.py build
cp -r build/lib* gate_lib