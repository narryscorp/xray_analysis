def load_data(train_path):

    def process_path(nb_class):
    
        def f(file_path):
            image = tf.io.read_file(file_path)    
            image = tf.image.decode_jpeg(image, channels=3)
            image = tf.image.convert_image_dtype(image, tf.float32)
         
            image = tf.image.resize(image, [127, 127], method='area')
            return image
        return f

    def reader_image(path_file, nb_class):

        list_ds = tf.data.Dataset.list_files(path_file)
        labeled_ds = list_ds.map(process_path(nb_class))
    
        return labeled_ds
    
#     train_ds = reader_image(train_path, 2)
#     val_ds = reader_image(val_path, batch_size, 2)

    print(type(train_ds))

testdata = reader_image("gs://qwiklabs-gcp-02-15ad15b6da61/chest_xray/val/NORMAL/NORMAL2-IM-1427-0001.jpeg")
print(testdata)
