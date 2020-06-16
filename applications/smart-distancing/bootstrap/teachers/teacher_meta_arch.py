import abc


class TeacherMetaArch(object):
    @abc.abstractmethod
    def inference(self, preprocessed_image):
        raise NotImplementedError

    @abc.abstractmethod
    def preprocessing(self, image):
        raise NotImplementedError

    @abc.abstractmethod
    def save_results(self, results):
        raise NotImplementedError
