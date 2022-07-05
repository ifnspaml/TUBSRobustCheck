from torch import nn


class ModelWrapper(nn.Module):
    """
    A class to wrap a model with a preprocess function as well as a postprocess function.
    """
    def __init__(self, model, preprocessing=None, postprocessing=None) -> None:
        """
        :param model: The model to be wrapped
        :param preprocess: A callable preprocess function
        :param postprocess: A callable postprocess function
        """

        super().__init__()

        # Initialization
        self.model = model
        self.preprocessing = preprocessing
        self.postprocessing = postprocessing

    def _preprocess(self, x):
        """
        A function to perform preprocessing on the model input.

        :param x: Input image to the model.
        :return: Preprocessed image
        """
        if self.preprocessing is None:
            return x
        else:
            return self.preprocessing(x)

    def _postprocess(self, y):
        """
        A function to perform postprocessing on the model output.

        :param y: Output of the model.
        :return: Postprocessed output
        """
        if self.postprocessing is None:
            return y
        else:
            return self.postprocessing(y)

    def forward(self, x):
        x = self._preprocess(x)
        y = self.model(x)
        return self._postprocess(y)
