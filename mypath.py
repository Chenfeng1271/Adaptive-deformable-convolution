class Path(object):
    @staticmethod
    def db_root_dir(dataset):
        if dataset == 'pascal':
            return 'E:\\dataset\\VOC\\VOCdevkit\\VOC2012'  # folder that contains VOCdevkit/.
        elif dataset == 'sbd':
            return 'E:\\dataset\\VOC\\benchmark_RELEASE'  # folder that contains dataset/.
        elif dataset == 'cityscapes':
            return 'E:\\dataset\\cityscapes'     # foler that contains leftImg8bit/
        elif dataset == 'coco':
            return '/path/to/datasets/coco/'
        else:
            print('Dataset {} not available.'.format(dataset))
            raise NotImplementedError
