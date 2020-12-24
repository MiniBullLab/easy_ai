# 代码打包
将python代码打包为一个可安装的库,具体打包过程如下：
（所有打包过程都在easy_py2so目录下进行）

#### easyai包打包
1. ``` cp -r ../easyai . ```
2. ``` python3 py2sec.py -d easyai -m __init__.py,setup.py```（该步骤执行时间会比较长几分钟）
3. ``` cd result```
4. 将ai_build目录中的全部文件拷贝到result目录中（如果需要打包对应版本的库，可以将最外层中的requirements_xxx文件的文件名修改为requirements，替换当前requirements文件）。
5. ``` python3 setup.py bdist_wheel```
6. 在文件夹dist中将whl文件拷贝走，即为最后打包好的文件
7. 将easy_py2so文件夹中生成的setup.py、build、result、tmp_build、log.log与easyai目录或文件删除

#### easy_convert包打包
1. ``` cp -r ../easy_converter . ```
2.  ``` rm -rf easy_converter/converter/tensorrt_utility ```
3.  ``` rm -rf easy_converter/converter/onnx_convert_tensorrt.py ```
4. ``` python3 py2sec.py -d easy_converter -m __init__.py,setup.py```（该步骤执行时间会比较长几分钟）
5. ``` cd result```
6. 将convert_build目录中的全部文件拷贝到result目录中（如果需要打包对应版本的库，可以将最外层中的requirements_xxx文件的文件名修改为requirements，替换当前requirements文件）。
7. ``` python3 setup.py bdist_wheel```
8. 在文件夹dist中将whl文件拷贝走，即为最后打包好的文件
9. 将easy_py2so文件夹中生成的setup.py、build、result、tmp_build、log.log与easy_converter目录或文件删除

#### easy_tools包打包
1. ``` cp -r ../easy_tools . ```
2. 在 http://118.31.19.101:8080/easy_tools/ 下载data压缩包，并解压到easy_tools目录下，并将解压的目录命名为data
3. ``` python3 py2sec.py -d easy_tools -m __init__.py,setup.py,easy_ai.py,easy_convert.py ```（该步骤执行时间会比较长几分钟）
4. ``` cd result```
5. 将tools_build目录中的全部文件拷贝到result目录中（如果需要打包对应版本的库，可以将最外层中的requirements_xxx文件的文件名修改为requirements，替换当前requirements文件）。
6. ``` python3 setup.py bdist_wheel```
6. 在文件夹dist中将whl文件拷贝走，即为最后打包好的文件
7. 将easy_py2so文件夹中生成的setup.py、build、result、tmp_build、log.log与easy_tools目录或文件删除
