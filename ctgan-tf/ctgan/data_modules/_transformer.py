"""
This module contains the definition of the DataTransformer.
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils._testing import ignore_warnings

from ..models import BGM, OHE


class DataTransformer:
    """Data Transformer.

    It models continuous columns with a :class:`ctgan.models.BGM` and
    normalizes to a scalar [0, 1] and a vector.

    Discrete columns are encoded using :class:`ctgan.model.OHE`.

    Parameters
    ----------
    n_clusters: int, default=10
        Number of modes.

    epsilon: float, default=0.005
        Epsilon value.

    Attributes
    ----------
    output_info: list[tuple]
        List containing metadata about the data columns of the original
        dataset, namely the number of columns where to apply a given activation
        function. Example: ``[(1, 'tanh', 1), (10, 'softmax', 1)]``

    output_dimensions: int
        Number of features generated by the transformer.

    output_tensor: list[tf.Tensor]
        List of `tf.Tensor` describing the metadata needed for correctly apply
        the mix of activation functions in :class:`ctgan.layers.GenActivation`.

    cond_tensor: list[tf.Tensor]
        List of `tf.Tensor` describing the metadata needed for correctly
        compute the conditional loss in :func:`ctgan.losses.conditional_loss`.
    """
    # pylint: disable=too-many-instance-attributes

    @classmethod
    def from_dict(cls, in_dict):
        """Create a new instance of this class by loading data from an
        external class dictionary.

        Parameters
        ----------
        in_dict: dict
            External class dictionary.

        Returns
        -------
        DataTransformer
            A new instance with the same internal data as the one
            provided by `in_dict`.
        """
        new_instance = DataTransformer()
        new_instance.__dict__ = in_dict
        return new_instance

    def __init__(self, n_clusters=10, epsilon=0.005):
        self._n_clusters = n_clusters
        self._epsilon = epsilon
        self._is_dataframe = None
        self._meta = None
        self._dtypes = None

        self.output_info = None
        self.output_dimensions = None
        self.output_tensor = None
        self.cond_tensor = None

    def generate_tensors(self):
        """Generates helper Tensorflow tensors, used in the CTGAN models.

        It generates two lists of tensors:

        - `output_tensor`: it is used in :class:`ctgan.layers.GenActivation`
          for applying a mix of activation functions (tanh or Gumbel-Softmax)
          according to the metadata extracted from the original dataset in
          :meth:`ctgan.data_modules.DataTransformer.fit`.
        - `cond_tensor`: it is used in :func:`ctgan.losses.conditional_loss`
          for computing the loss of the conditional Generator,
          according to the metadata extracted from the original dataset in
          :meth:`ctgan.data_modules.DataTransformer.fit`.

        """
        if self.output_info is None:
            raise AttributeError("Output info still not available")

        output_info = []
        cond_info = []
        i = 0
        st_idx = 0
        st_c = 0
        for item in self.output_info:
            ed_idx = st_idx + item[0]
            if not item[2]:
                ed_c = st_c + item[0]
                cond_info.append(tf.constant(
                    [st_idx, ed_idx, st_c, ed_c, i], dtype=tf.int32))
                st_c = ed_c
                i += 1

            output_info.append(tf.constant(
                [st_idx, ed_idx, int(item[1] == 'softmax')], dtype=tf.int32))
            st_idx = ed_idx

        self.output_tensor = output_info
        self.cond_tensor = cond_info

    @ignore_warnings(category=ConvergenceWarning)
    def _fit_continuous(self, column, data):
        """Fits a Variational Gaussian Mixture Model to the corresponding data
        column, and returns a dictionary describing the obtained model.

        Parameters
        ----------
        column: str
            Description of the data column.

        data: np.ndarray
            Data column.

        Returns
        -------
        dict
            A dictionary containing:

            - `name`: the column description.
            - `model`: the VGM model.
            - `components`: the number of Gaussian components
            - `output_info`: a list indicating which activation functions
              should be applied to the outputted columns (tanh - 1 for
              :math:`\\alpha`, softmax - number of components for
              :math:`\\beta`)
              Example: ``[(1, 'tanh', 1), (num_components, 'softmax', 1)]``
            - `output_dimensions`: the total number of outputs
              (1 + number of components).

        """
        vgm = BGM(
            self._n_clusters,
            weight_concentration_prior_type='dirichlet_process',
            weight_concentration_prior=0.001,
            n_init=1
        )
        vgm.fit(data)
        components = vgm.weights_ > self._epsilon
        num_components = components.sum()

        return {
            'name': column,
            'model': vgm,
            'components': components,
            'output_info': [(1, 'tanh', 1), (num_components, 'softmax', 1)],
            'output_dimensions': 1 + num_components,
        }

    def _fit_discrete(self, column, data):
        """Fits an One-Hot encoder to the corresponding data column,
        and returns a dictionary describing the obtained transformer.

        Parameters
        ----------
        column: str
            Description of the data column.

        data: np.ndarray
            Data column.

        Returns
        -------
        dict
            A dictionary containing:

            - `name`: the column description.
            - `encoder`: the OneHot encoder.
            - `output_info`: a list indicating which activation functions
              should be applied to the outputted columns (softmax - number of
              categories). Example: ``[(categories, 'softmax', 0)]``
            - `output_dimensions`: the total number of categories.

        """
        ohe = OHE(sparse=False)
        ohe.fit(data)
        categories = len(ohe.categories_[0])

        return {
            'name': column,
            'encoder': ohe,
            'output_info': [(categories, 'softmax', 0)],
            'output_dimensions': categories
        }

    def fit(self, data, discrete_columns=tuple()):
        """Fits the data transformer to the input data, according to the
        passed discrete_columns` list.

        For each continuous column :math:`C_i`, it will use variational
        Gaussian mixture model (VGM) to estimate the number of modes
        :math:`m_i` and fit a Gaussian mixture.

        For each discrete column :math:`d_i`, it will use an One Hot encoder
        to estimate the number of categories in the data.

        After transforming each one of the data columns, it gathers all the
        information gathered from
        :meth:`ctgan.data_modules.DataTransformer._fit_continuous` and
        :meth:`ctgan.data_modules.DataTransformer._fit_discrete` and
        saves it into the current instance.

        Parameters
        ----------
        data: np.ndarray, or pandas.DataFrame
            Input dataset.

        discrete_columns: list[str]
            List containing which data columns will be considered discrete
            variables.
        """

        self.output_info = []
        self.output_dimensions = 0

        if not isinstance(data, pd.DataFrame):
            self._is_dataframe = False
            data = pd.DataFrame(data)
        else:
            self._is_dataframe = True

        self._dtypes = data.infer_objects().dtypes
        self._meta = []
        for column in data.columns:
            column_data = data[[column]].values
            if column in discrete_columns:
                meta = self._fit_discrete(column, column_data)
            else:
                meta = self._fit_continuous(column, column_data)

            self.output_info += meta['output_info']
            self.output_dimensions += meta['output_dimensions']
            self._meta.append(meta)

    def _transform_continuous(self, column_meta, data):
        """Transforms a given column of the input data according to the
        corresponding VGM model.

        Parameters
        ----------
        column_meta: dict
            A dictionary containing column metadata as described in the output
            of :meth:`ctgan.data_modules.DataTransformer._fit_continuous`.

        data: np.ndarray
            Input data column to be transformed.

        Returns
        -------
        list[np.ndarray, np.ndarray]
            A list containing the transformed features, and the probability of
            each feature point belonging to a given mode on the VGM model.

        """
        components = column_meta['components']
        model = column_meta['model']

        means = model.means_.reshape((1, self._n_clusters))
        stds = np.sqrt(model.covariances_).reshape((1, self._n_clusters))
        features = (data - means) / (4 * stds)

        probs = model.predict_proba(data)

        n_opts = components.sum()
        features = features[:, components]
        probs = probs[:, components]

        opt_sel = np.zeros(len(data), dtype='int')
        for i in range(len(data)):
            norm_probs = probs[i] + 1e-6
            norm_probs = norm_probs / norm_probs.sum()
            opt_sel[i] = np.random.choice(np.arange(n_opts), p=norm_probs)

        idx = np.arange((len(features)))
        features = features[idx, opt_sel].reshape([-1, 1])
        features = np.clip(features, -.99, .99)

        probs_onehot = np.zeros_like(probs)
        probs_onehot[np.arange(len(probs)), opt_sel] = 1
        return [features, probs_onehot]

    def _transform_discrete(self, column_meta, data):
        """Encodes a given discrete column to OneHot format.

        Parameters
        ----------
        column_meta: dict
            A dictionary containing column metadata as described in the output
            of :meth:`ctgan.data_modules.DataTransformer._fit_discrete`.

        data: np.ndarray
            Input data column to be transformed.

        Returns
        -------
        np.ndarray
            OneHot encoded form of the input data column.
        """
        encoder = column_meta['encoder']
        return encoder.transform(data)

    def transform(self, data):
        """Transform the input data according to the metadata obtained from
        data fitting in :meth:`ctgan.data_modules.DataTransformer.fit`.

        Parameters
        ----------
        data: np.ndarray, or pandas.DataFrame
            Input data to be transformed.

        Returns
        -------
        np.ndarray
            Transformed input data.
        """
        if not isinstance(data, pd.DataFrame):
            data = pd.DataFrame(data)

        values = []
        for meta in self._meta:
            column_data = data[[meta['name']]].values
            if 'model' in meta:
                values += self._transform_continuous(meta, column_data)
            else:
                values.append(self._transform_discrete(meta, column_data))

        return np.concatenate(values, axis=1).astype(float)

    def _inverse_transform_continuous(self, meta, data, sigma=None):
        """Inverse transforms continuous data.

        The method takes the metadata from a given continuous column of the
        original dataset and transforms the data outputted by the
        :class:`ctgan.models.Generator` to the original format.

        Parameters
        ----------
        meta: dict
            A dictionary containing column metadata as described in the output
            of :meth:`ctgan.data_modules.DataTransformer._fit_continuous`.

        data:
            Transformed data column features.
        sigma: float, default=None

        Returns
        -------
        np.ndarray
            A data column in the appropriate original format.
        """
        model = meta['model']
        components = meta['components']

        mean = data[:, 0]
        variance = data[:, 1:]

        if sigma is not None:
            mean = np.random.normal(mean, sigma)

        mean = np.clip(mean, -1, 1)
        v_t = np.ones((len(data), self._n_clusters)) * -100
        v_t[:, components] = variance
        variance = v_t
        means = model.means_.reshape([-1])
        stds = np.sqrt(model.covariances_).reshape([-1])
        p_argmax = np.argmax(variance, axis=1)
        std_t = stds[p_argmax]
        mean_t = means[p_argmax]
        column = mean * 4 * std_t + mean_t

        return column

    def _inverse_transform_discrete(self, meta, data):
        """Inverse transforms discrete data.

        The method takes the metadata from a given discrete column of the
        original dataset and transforms the data outputted by the
        :class:`ctgan.models.Generator` to the original format.

        Parameters
        ----------
        meta: dict
            A dictionary containing column metadata as described in the output
            of :meth:`ctgan.data_modules.DataTransformer._fit_discrete`.

        data:
            Transformed data column features.

        Returns
        -------
        np.ndarray
            A data column in the appropriate original format.
        """
        encoder = meta['encoder']
        return encoder.inverse_transform(data)

    def inverse_transform(self, data, sigmas=None):
        """Inverse transforms data outputted by
        :class:`ctgan.models.Generator`, according to the metadata obtained
        from the original dataset in
        :meth:`ctgan.data_modules.DataTransformer.fit`.

        Parameters
        ----------
        data: np.ndarray
            Data outputted by :class:`ctgan.models.Generator`, that will be
            transformed to the same format as the original dataset.

        sigmas: list[float], default=None

        Returns
        -------
        np.ndarray, or pandas.DataFrame
            Returns an `np.ndarray`, or a `pandas.DataFrame` if the original
            dataset was provided as one, containing the synthetic dataset
            with the same format as the original dataset.
        """
        start = 0
        output = []
        column_names = []
        for meta in self._meta:
            dimensions = meta['output_dimensions']
            columns_data = data[:, start:start + dimensions]

            if 'model' in meta:
                sigma = sigmas[start] if sigmas else None
                inverted = self._inverse_transform_continuous(
                    meta, columns_data, sigma)
            else:
                inverted = self._inverse_transform_discrete(meta, columns_data)

            output.append(inverted)
            column_names.append(meta['name'])
            start += dimensions

        output = np.column_stack(output)
        output = pd.DataFrame(output, columns=column_names)\
            .astype(self._dtypes)
        if not self._is_dataframe:
            output = output.values

        return output
