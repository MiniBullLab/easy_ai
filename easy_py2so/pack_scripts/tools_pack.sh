cp -r ../easy_tools .

echo "Downloading data......"
wget http://118.31.19.101:8080/easy_tools/data_v3.zip -O ./easy_tools/data.zip
unzip -o ./easy_tools/data.zip -d ./easy_tools/
rm -rf ./easy_tools/data.zip

echo "Waiting for packing......"
python3 py2sec.py -d easy_tools -m __init__.py,setup.py,easy_ai.py,easy_convert.py

cd result
cp ../tools_build/* .
python3 setup.py bdist_wheel
cp dist/*.whl ../
cd ..

rm -rf setup.py build result tmp_build log.log easy_tools
echo "Pack Done!"