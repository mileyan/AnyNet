rm -rf build
rm -rf gate_lib
python3 setup.py clean
python3 setup.py build
cp -r build/lib* gate_lib
