cp -r ../easy_converter .

rm -rf easy_converter/converter/tensorrt_utility
rm -rf easy_converter/converter/onnx_convert_tensorrt.py
echo "Waiting for packing......"
python3 py2sec.py -d easy_converter -m __init__.py,setup.py

cd result
cp ../convert_build/* .
python3 setup.py bdist_wheel
cp dist/*.whl ../
cd ..

rm -rf setup.py build result tmp_build log.log easy_converter
echo "Pack Done!"