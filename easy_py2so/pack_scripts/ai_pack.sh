cp -r ../easyai .

echo "Waiting for packing......"
python3 py2sec.py -d easyai -m __init__.py,setup.py,flops_counter.py

cd result
cp ../ai_build/* .
python3 setup.py bdist_wheel
cp dist/*.whl ../
cd ..

rm -rf setup.py build result tmp_build log.log easyai
echo "Pack Done!"