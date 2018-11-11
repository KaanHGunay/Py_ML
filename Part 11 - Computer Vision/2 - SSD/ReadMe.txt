Burada ki kodları kullanmak için;
1 - Tensorflow'u githubdan indir (https://github.com/tensorflow/models)
2 - Protobuf'ı indir - 3.4 sürümü sorunsuz çalıştı (https://github.com/protocolbuffers/protobuf/releases)
3 - Protobuf ile \models\research\object_detection içinde ki protos klasöründeki proto uzantılı dosyaları derle
    (protoc object_detection/protos/*.proto --python_out=.)
4 - \models\research\object_detection klasöründe çalış