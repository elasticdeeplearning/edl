#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import collections
import six
import sys
from functools import partial, reduce

import numpy as np

import paddle
import paddle.fluid as fluid
import paddle.fluid.layers.utils as utils
from paddle.fluid import layers
from paddle.fluid.layers import BeamSearchDecoder
from paddle.fluid.layers.utils import map_structure, flatten, pack_sequence_as
from paddle.fluid.dygraph import Layer, Embedding, Linear, LayerNorm, GRUUnit, Conv2D, Pool2D
from paddle.fluid.data_feeder import convert_dtype


class RNNCell(Layer):
    """
    RNNCell is the base class for abstraction representing the calculations
    mapping the input and state to the output and new state. It is suitable to
    and mostly used in RNN.
    """

    def get_initial_states(self,
                           batch_ref,
                           shape=None,
                           dtype=None,
                           init_value=0,
                           batch_dim_idx=0):
        """
        Generate initialized states according to provided shape, data type and
        value.

        Parameters:
            batch_ref: A (possibly nested structure of) tensor variable[s].
                The first dimension of the tensor will be used as batch size to
                initialize states.
            shape: A (possibly nested structure of) shape[s], where a shape is
                represented as a list/tuple of integer). -1(for batch size) will
                beautomatically inserted if shape is not started with it. If None,
                property `state_shape` will be used. The default value is None.
            dtype: A (possibly nested structure of) data type[s]. The structure
                must be same as that of `shape`, except when all tensors' in states
                has the same data type, a single data type can be used. If None and
                property `cell.state_shape` is not available, float32 will be used
                as the data type. The default value is None.
            init_value: A float value used to initialize states.
            batch_dim_idx: An integer indicating which dimension of the tensor in
                inputs represents batch size.  The default value is 0.

        Returns:
            Variable: tensor variable[s] packed in the same structure provided \
                by shape, representing the initialized states.
        """
        # TODO: use inputs and batch_size
        batch_ref = flatten(batch_ref)[0]

        def _is_shape_sequence(seq):
            if sys.version_info < (3, ):
                integer_types = (
                    int,
                    long, )
            else:
                integer_types = (int, )
            """For shape, list/tuple of integer is the finest-grained objection"""
            if (isinstance(seq, list) or isinstance(seq, tuple)):
                if reduce(
                        lambda flag, x: isinstance(x, integer_types) and flag,
                        seq, True):
                    return False
            # TODO: Add check for the illegal
            if isinstance(seq, dict):
                return True
            return (isinstance(seq, collections.Sequence) and
                    not isinstance(seq, six.string_types))

        class Shape(object):
            def __init__(self, shape):
                self.shape = shape if shape[0] == -1 else ([-1] + list(shape))

        # nested structure of shapes
        states_shapes = self.state_shape if shape is None else shape
        is_sequence_ori = utils.is_sequence
        utils.is_sequence = _is_shape_sequence
        states_shapes = map_structure(lambda shape: Shape(shape),
                                      states_shapes)
        utils.is_sequence = is_sequence_ori

        # nested structure of dtypes
        try:
            states_dtypes = self.state_dtype if dtype is None else dtype
        except NotImplementedError:  # use fp32 as default
            states_dtypes = "float32"
        if len(flatten(states_dtypes)) == 1:
            dtype = flatten(states_dtypes)[0]
            states_dtypes = map_structure(lambda shape: dtype, states_shapes)

        init_states = map_structure(
            lambda shape, dtype: fluid.layers.fill_constant_batch_size_like(
                input=batch_ref,
                shape=shape.shape,
                dtype=dtype,
                value=init_value,
                input_dim_idx=batch_dim_idx), states_shapes, states_dtypes)
        return init_states

    @property
    def state_shape(self):
        """
        Abstract method (property).
        Used to initialize states.
        A (possiblely nested structure of) shape[s], where a shape is represented
        as a list/tuple of integers (-1 for batch size would be automatically
        inserted into a shape if shape is not started with it).
        Not necessary to be implemented if states are not initialized by
        `get_initial_states` or the `shape` argument is provided when using
        `get_initial_states`.
        """
        raise NotImplementedError(
            "Please add implementaion for `state_shape` in the used cell.")

    @property
    def state_dtype(self):
        """
        Abstract method (property).
        Used to initialize states.
        A (possiblely nested structure of) data types[s]. The structure must be
        same as that of `shape`, except when all tensors' in states has the same
        data type, a signle data type can be used.
        Not necessary to be implemented if states are not initialized
        by `get_initial_states` or the `dtype` argument is provided when using
        `get_initial_states`.
        """
        raise NotImplementedError(
            "Please add implementaion for `state_dtype` in the used cell.")


class BasicLSTMCell(RNNCell):
    """
    Long-Short Term Memory(LSTM) RNN cell.

    The formula used is as follows:

    .. math::

        i_{t} & = act_g(W_{x_{i}}x_{t} + W_{h_{i}}h_{t-1} + b_{i})

        f_{t} & = act_g(W_{x_{f}}x_{t} + W_{h_{f}}h_{t-1} + b_{f} + forget\\_bias)

        c_{t} & = f_{t}c_{t-1} + i_{t} act_c (W_{x_{c}}x_{t} + W_{h_{c}}h_{t-1} + b_{c})

        o_{t} & = act_g(W_{x_{o}}x_{t} + W_{h_{o}}h_{t-1} + b_{o})

        h_{t} & = o_{t} act_c (c_{t})

    Please refer to `An Empirical Exploration of Recurrent Network Architectures
    <http://proceedings.mlr.press/v37/jozefowicz15.pdf>`_ for more details.

    Parameters:
        input_size (int): The input size in the LSTM cell.
        hidden_size (int): The hidden size in the LSTM cell.
        param_attr(ParamAttr, optional): The parameter attribute for the learnable
            weight matrix. Default: None.
        bias_attr (ParamAttr, optional): The parameter attribute for the bias
            of LSTM. Default: None.
        gate_activation (function, optional): The activation function for gates
            of LSTM, that is :math:`act_g` in the formula. Default: None,
            representing for `fluid.layers.sigmoid`.
        activation (function, optional): The non-gate activation function of
            LSTM, that is :math:`act_c` in the formula. Default: None,
            representing for 'fluid.layers.tanh'.
        forget_bias(float, optional): forget bias used when computing forget gate.
            Default 1.0
        dtype(string, optional): The data type used in this cell. Default float32.

    Examples:

        .. code-block:: python

            import paddle
            import paddle.fluid as fluid
            from paddle.incubate.hapi.text import BasicLSTMCell, RNN

            inputs = paddle.rand((2, 4, 32))
            cell = BasicLSTMCell(input_size=32, hidden_size=64)
            rnn = RNN(cell=cell)
            outputs, _ = rnn(inputs)  # [2, 4, 64]
    """

    def __init__(self,
                 input_size,
                 hidden_size,
                 param_attr=None,
                 bias_attr=None,
                 gate_activation=None,
                 activation=None,
                 forget_bias=1.0,
                 dtype='float32'):
        super(BasicLSTMCell, self).__init__()

        self._hidden_size = hidden_size
        self._param_attr = param_attr
        self._bias_attr = bias_attr
        self._gate_activation = gate_activation or layers.sigmoid
        self._activation = activation or layers.tanh
        # TODO(guosheng): find better way to resolve constants in __init__
        self._forget_bias = layers.create_global_var(
            shape=[1], dtype=dtype, value=forget_bias, persistable=True)
        # TODO(guosheng): refine this if recurrent_op removes gradient require
        self._forget_bias.stop_gradient = False
        self._dtype = dtype
        self._input_size = input_size

        self._weight = self.create_parameter(
            attr=self._param_attr,
            shape=[
                self._input_size + self._hidden_size, 4 * self._hidden_size
            ],
            dtype=self._dtype)

        self._bias = self.create_parameter(
            attr=self._bias_attr,
            shape=[4 * self._hidden_size],
            dtype=self._dtype,
            is_bias=True)

    def forward(self, inputs, states):
        """
        Performs single step LSTM calculations.

        Parameters:
            inputs (Variable): A tensor with shape `[batch_size, input_size]`,
                corresponding to :math:`x_t` in the formula. The data type
                should be float32 or float64.
            states (Variable): A list of containing two tensors, each shaped
                `[batch_size, hidden_size]`, corresponding to :math:`h_{t-1}, c_{t-1}`
                in the formula. The data type should be float32 or float64.

        Returns:
            tuple: A tuple( :code:`(outputs, new_states)` ), where `outputs` is \
                a tensor with shape `[batch_size, hidden_size]`, corresponding \
                to :math:`h_{t}` in the formula; `new_states` is a list containing \
                two tenser variables shaped `[batch_size, hidden_size]`, corresponding \
                to :math:`h_{t}, c_{t}` in the formula. The data type of these \
                tensors all is same as that of `states`.
        """
        pre_hidden, pre_cell = states
        concat_input_hidden = layers.concat([inputs, pre_hidden], 1)
        gate_input = layers.matmul(x=concat_input_hidden, y=self._weight)
        gate_input = layers.elementwise_add(gate_input, self._bias)
        i, j, f, o = layers.split(gate_input, num_or_sections=4, dim=-1)
        new_cell = layers.elementwise_add(
            layers.elementwise_mul(
                pre_cell,
                self._gate_activation(
                    layers.elementwise_add(f, self._forget_bias))),
            layers.elementwise_mul(
                self._gate_activation(i), self._activation(j)))
        new_hidden = self._activation(new_cell) * self._gate_activation(o)

        return new_hidden, [new_hidden, new_cell]

    @property
    def state_shape(self):
        """
        The `state_shape` of BasicLSTMCell is a list with two shapes: `[[hidden_size], [hidden_size]]`
        (-1 for batch size would be automatically inserted into shape). These two
        shapes correspond to :math:`h_{t-1}` and :math:`c_{t-1}` separately.
        """
        return [[self._hidden_size], [self._hidden_size]]


class BasicGRUCell(RNNCell):
    """
    Gated Recurrent Unit (GRU) RNN cell.

    The formula for GRU used is as follows:

    .. math::

        u_t & = act_g(W_{ux}x_{t} + W_{uh}h_{t-1} + b_u)

        r_t & = act_g(W_{rx}x_{t} + W_{rh}h_{t-1} + b_r)

        \\tilde{h_t} & = act_c(W_{cx}x_{t} + W_{ch}(r_t \odot h_{t-1}) + b_c)

        h_t & = u_t \odot h_{t-1} + (1-u_t) \odot \\tilde{h_t}

    Please refer to `An Empirical Exploration of Recurrent Network Architectures
    <http://proceedings.mlr.press/v37/jozefowicz15.pdf>`_ for more details.

    Parameters:
        input_size (int): The input size for the first GRU cell.
        hidden_size (int): The hidden size for every GRU cell.
        param_attr(ParamAttr, optional): The parameter attribute for the learnable
            weight matrix. Default: None.
        bias_attr (ParamAttr, optional): The parameter attribute for the bias
            of LSTM. Default: None.
        gate_activation (function, optional): The activation function for gates
            of GRU, that is :math:`act_g` in the formula. Default: None,
            representing for `fluid.layers.sigmoid`.
        activation (function, optional): The non-gate activation function of
            GRU, that is :math:`act_c` in the formula. Default: None,
            representing for 'fluid.layers.tanh'.
        dtype(string, optional): The data type used in this cell. Default float32.

    Examples:

        .. code-block:: python

            import paddle
            import paddle.fluid as fluid
            from paddle.incubate.hapi.text import BasicGRUCell, RNN

            inputs = paddle.rand((2, 4, 32))
            cell = BasicGRUCell(input_size=32, hidden_size=64)
            rnn = RNN(cell=cell)
            outputs, _ = rnn(inputs)  # [2, 4, 64]
    """

    def __init__(self,
                 input_size,
                 hidden_size,
                 param_attr=None,
                 bias_attr=None,
                 gate_activation=None,
                 activation=None,
                 dtype='float32'):
        super(BasicGRUCell, self).__init__()
        self._input_size = input_size
        self._hidden_size = hidden_size
        self._param_attr = param_attr
        self._bias_attr = bias_attr
        self._gate_activation = gate_activation or layers.sigmoid
        self._activation = activation or layers.tanh
        self._dtype = dtype

        if self._param_attr is not None and self._param_attr.name is not None:
            gate_param_attr = copy.deepcopy(self._param_attr)
            candidate_param_attr = copy.deepcopy(self._param_attr)
            gate_param_attr.name += "_gate"
            candidate_param_attr.name += "_candidate"
        else:
            gate_param_attr = self._param_attr
            candidate_param_attr = self._param_attr

        self._gate_weight = self.create_parameter(
            attr=gate_param_attr,
            shape=[
                self._input_size + self._hidden_size, 2 * self._hidden_size
            ],
            dtype=self._dtype)

        self._candidate_weight = self.create_parameter(
            attr=candidate_param_attr,
            shape=[self._input_size + self._hidden_size, self._hidden_size],
            dtype=self._dtype)

        if self._bias_attr is not None and self._bias_attr.name is not None:
            gate_bias_attr = copy.deepcopy(self._bias_attr)
            candidate_bias_attr = copy.deepcopy(self._bias_attr)
            gate_bias_attr.name += "_gate"
            candidate_bias_attr.name += "_candidate"
        else:
            gate_bias_attr = self._bias_attr
            candidate_bias_attr = self._bias_attr

        self._gate_bias = self.create_parameter(
            attr=gate_bias_attr,
            shape=[2 * self._hidden_size],
            dtype=self._dtype,
            is_bias=True)
        self._candidate_bias = self.create_parameter(
            attr=candidate_bias_attr,
            shape=[self._hidden_size],
            dtype=self._dtype,
            is_bias=True)

    def forward(self, inputs, states):
        """
        Performs single step GRU calculations.

        Parameters:
            inputs (Variable): A tensor with shape `[batch_size, input_size]`,
                corresponding to :math:`x_t` in the formula. The data type
                should be float32 or float64.
            states (Variable): A tensor with shape `[batch_size, hidden_size]`.
                corresponding to :math:`h_{t-1}` in the formula. The data type
                should be float32 or float64.

        Returns:
            tuple: A tuple( :code:`(outputs, new_states)` ), where `outputs` and \
                `new_states` is the same tensor shaped `[batch_size, hidden_size]`, \
                corresponding to :math:`h_t` in the formula. The data type of the \
                tensor is same as that of `states`.        
        """
        pre_hidden = states
        concat_input_hidden = layers.concat([inputs, pre_hidden], axis=1)

        gate_input = layers.matmul(x=concat_input_hidden, y=self._gate_weight)

        gate_input = layers.elementwise_add(gate_input, self._gate_bias)

        gate_input = self._gate_activation(gate_input)
        r, u = layers.split(gate_input, num_or_sections=2, dim=1)

        r_hidden = r * pre_hidden

        candidate = layers.matmul(
            layers.concat([inputs, r_hidden], 1), self._candidate_weight)
        candidate = layers.elementwise_add(candidate, self._candidate_bias)

        c = self._activation(candidate)
        new_hidden = u * pre_hidden + (1 - u) * c

        return new_hidden, new_hidden

    @property
    def state_shape(self):
        """
        The `state_shape` of BasicGRUCell is a shape `[hidden_size]` (-1 for batch
        size would be automatically inserted into shape). The shape corresponds
        to :math:`h_{t-1}`.
        """
        return [self._hidden_size]


class RNN(Layer):
    """
    RNN creates a recurrent neural network specified by RNNCell `cell`, which
    performs :code:`cell.forward()` repeatedly until reaches to the maximum
    length of `inputs`.

    Parameters:
        cell(RNNCell): An instance of `RNNCell`.
        is_reverse (bool, optional): Indicate whether to calculate in the reverse
            order of input sequences. Default: `False`.
        time_major (bool, optional): Indicate the data layout of Tensor included
            in `input` and `output` tensors. If `False`, the data layout would
            be batch major with shape `[batch_size, sequence_length, ...]`.  If
            `True`, the data layout would be time major with shape
            `[sequence_length, batch_size, ...]`. Default: `False`.

    Examples:

        .. code-block:: python

            import paddle
            import paddle.fluid as fluid
            from paddle.incubate.hapi.text import StackedLSTMCell, RNN

            inputs = paddle.rand((2, 4, 32))
            cell = StackedLSTMCell(input_size=32, hidden_size=64)
            rnn = RNN(cell=cell)
            outputs, _ = rnn(inputs)  # [2, 4, 64]
    """

    def __init__(self, cell, is_reverse=False, time_major=False):
        super(RNN, self).__init__()
        self.cell = cell
        if not hasattr(self.cell, "call"):
            self.cell.call = self.cell.forward
        self.is_reverse = is_reverse
        self.time_major = time_major
        self.batch_index, self.time_step_index = (1, 0) if time_major else (0,
                                                                            1)

    def forward(self,
                inputs,
                initial_states=None,
                sequence_length=None,
                **kwargs):
        """
        Performs :code:`cell.forward()` repeatedly until reaches to the maximum
        length of `inputs`.

        Parameters:
            inputs (Variable): A (possibly nested structure of) tensor variable[s]. 
                The shape of tensor should be `[batch_size, sequence_length, ...]`
                for `time_major == False` or `[sequence_length, batch_size, ...]`
                for `time_major == True`. It represents the inputs to be unrolled
                in RNN.
            initial_states (Variable, optional): A (possibly nested structure of)
                tensor variable[s], representing the initial state for RNN. 
                If not provided, `cell.get_initial_states` would be used to produce
                the initial state. Default None.
            sequence_length (Variable, optional): A tensor with shape `[batch_size]`.
                It stores real length of each instance, thus enables users to extract
                the last valid state when past a batch element's sequence length for
                correctness. If not provided, the paddings would be treated same as
                non-padding inputs. Default None.
            **kwargs: Additional keyword arguments. Arguments passed to `cell.forward`. 

        Returns:
            tuple: A tuple( :code:`(final_outputs, final_states)` ) including the final \
                outputs and states, both are Tensor or nested structure of Tensor. \
                `final_outputs` has the same structure and data types as \
                the returned `outputs` of :code:`cell.forward` , and each Tenser in `final_outputs` \
                stacks all time steps' counterpart in `outputs` thus has shape `[batch_size, sequence_length, ...]` \
                for `time_major == False` or `[sequence_length, batch_size, ...]` for `time_major == True`. \
                `final_states` is the counterpart at last time step of initial states, \
                thus has the same structure with it and has tensors with same shapes \
                and data types.
        """
        if fluid.in_dygraph_mode():

            class ArrayWrapper(object):
                def __init__(self, x):
                    self.array = [x]

                def append(self, x):
                    self.array.append(x)
                    return self

            def _maybe_copy(state, new_state, step_mask):
                # TODO: use where_op
                new_state = fluid.layers.elementwise_mul(
                    new_state, step_mask,
                    axis=0) - fluid.layers.elementwise_mul(
                        state, (step_mask - 1), axis=0)
                return new_state

            flat_inputs = flatten(inputs)
            batch_size, time_steps = (
                flat_inputs[0].shape[self.batch_index],
                flat_inputs[0].shape[self.time_step_index])

            if initial_states is None:
                initial_states = self.cell.get_initial_states(
                    batch_ref=inputs, batch_dim_idx=self.batch_index)

            if not self.time_major:
                inputs = map_structure(
                    lambda x: fluid.layers.transpose(x, [1, 0] + list(
                        range(2, len(x.shape)))), inputs)

            if sequence_length is not None:
                mask = fluid.layers.sequence_mask(
                    sequence_length,
                    maxlen=time_steps,
                    dtype=flatten(initial_states)[0].dtype)
                mask = fluid.layers.transpose(mask, [1, 0])

            if self.is_reverse:
                inputs = map_structure(
                    lambda x: fluid.layers.reverse(x, axis=[0]), inputs)
                mask = fluid.layers.reverse(
                    mask, axis=[0]) if sequence_length is not None else None

            states = initial_states
            outputs = []
            for i in range(time_steps):
                step_inputs = map_structure(lambda x: x[i], inputs)
                step_outputs, new_states = self.cell(step_inputs, states,
                                                     **kwargs)
                if sequence_length is not None:
                    new_states = map_structure(
                        partial(
                            _maybe_copy, step_mask=mask[i]),
                        states,
                        new_states)
                states = new_states
                outputs = map_structure(
                    lambda x: ArrayWrapper(x),
                    step_outputs) if i == 0 else map_structure(
                        lambda x, x_array: x_array.append(x), step_outputs,
                        outputs)

            final_outputs = map_structure(
                lambda x: fluid.layers.stack(x.array, axis=self.time_step_index
                                             ), outputs)

            if self.is_reverse:
                final_outputs = map_structure(
                    lambda x: fluid.layers.reverse(x, axis=self.time_step_index
                                                   ), final_outputs)

            final_states = new_states
        else:
            final_outputs, final_states = fluid.layers.rnn(
                self.cell,
                inputs,
                initial_states=initial_states,
                sequence_length=sequence_length,
                time_major=self.time_major,
                is_reverse=self.is_reverse,
                **kwargs)
        return final_outputs, final_states


class StackedRNNCell(RNNCell):
    """
    Wrapper allowing a stack of RNN cells to behave as a single cell. It is used
    to implement stacked RNNs.

    Parameters:
        cells (list|tuple): List of RNN cell instances.

    Examples:

        .. code-block:: python

            from paddle.incubate.hapi.text import BasicLSTMCell, StackedRNNCell

            cells = [BasicLSTMCell(32, 32), BasicLSTMCell(32, 32)]
            stack_rnn = StackedRNNCell(cells)
    """

    def __init__(self, cells):
        super(StackedRNNCell, self).__init__()
        self.cells = []
        for i, cell in enumerate(cells):
            self.cells.append(self.add_sublayer("cell_%d" % i, cell))

    def forward(self, inputs, states, **kwargs):
        """
        Performs :code:`cell.forward` for all including cells sequentially.
        Each cell's `inputs` is the `outputs` of the previous cell. And each
        cell's `states` is the corresponding one in `states`.

        Parameters:
            inputs (Variable): The inputs for the first cell. Mostly it is a
                float32 or float64 tensor with shape `[batch_size, input_size]`.
            states (list): A list containing states for all cells orderly.
            **kwargs: Additional keyword arguments, which passed to `cell.forward`
                for all including cells.

        Returns:
            tuple: A tuple( :code:`(outputs, new_states)` ). `outputs` is the \
                `outputs` of the last cell. `new_states` is a list composed \
                of all cells' `new_states`, and its structure and data type is \
                same as that of `states` argument.
        """
        new_states = []
        for cell, state in zip(self.cells, states):
            outputs, new_state = cell(inputs, state, **kwargs)
            inputs = outputs
            new_states.append(new_state)
        return outputs, new_states

    @staticmethod
    def stack_param_attr(param_attr, n):
        """
        If `param_attr` is a list or tuple, convert every element in it to a
        ParamAttr instance. Otherwise, repeat `param_attr` `n` times to
        construct a list, and rename every one by appending a increasing index
        suffix to avoid having same names when `param_attr` contains a name.

        Parameters:
            param_attr (list|tuple|ParamAttr): A list, tuple or something can be
                converted to a ParamAttr instance by `ParamAttr._to_attr`.
            n (int): The times to repeat to construct a list when `param_attr`
                is not a list or tuple.

        Returns:
            list: A list composed of each including cell's `param_attr`.
        """
        if isinstance(param_attr, (list, tuple)):
            assert len(param_attr) == n, (
                "length of param_attr should be %d when it is a list/tuple" %
                n)
            param_attrs = [
                fluid.ParamAttr._to_attr(attr) for attr in param_attr
            ]
        else:
            param_attrs = []
            attr = fluid.ParamAttr._to_attr(param_attr)
            for i in range(n):
                attr_i = copy.deepcopy(attr)
                if attr.name:
                    attr_i.name = attr_i.name + "_" + str(i)
                param_attrs.append(attr_i)
        return param_attrs

    @property
    def state_shape(self):
        """
        The `state_shape` of StackedRNNCell is a list composed of each including
        cell's `state_shape`.

        Returns:
            list: A list composed of each including cell's `state_shape`.
        """
        return [cell.state_shape for cell in self.cells]


class StackedLSTMCell(RNNCell):
    """
    Wrapper allowing a stack of LSTM cells to behave as a single cell. It is used
    to implement stacked LSTM.

    The formula for LSTM used here is as follows:

    .. math::

        i_{t} & = act_g(W_{x_{i}}x_{t} + W_{h_{i}}h_{t-1} + b_{i})

        f_{t} & = act_g(W_{x_{f}}x_{t} + W_{h_{f}}h_{t-1} + b_{f} + forget\\_bias)

        c_{t} & = f_{t}c_{t-1} + i_{t} act_c (W_{x_{c}}x_{t} + W_{h_{c}}h_{t-1} + b_{c})

        o_{t} & = act_g(W_{x_{o}}x_{t} + W_{h_{o}}h_{t-1} + b_{o})

        h_{t} & = o_{t} act_c (c_{t})


    Parameters:
        input_size (int): The input size for the first LSTM cell.
        hidden_size (int): The hidden size for every LSTM cell.
        gate_activation (function, optional): The activation function for gates
            of LSTM, that is :math:`act_g` in the formula. Default: None,
            representing for `fluid.layers.sigmoid`.
        activation (function, optional): The non-gate activation function of
            LSTM, that is :math:`act_c` in the formula. Default: None,
            representing for 'fluid.layers.tanh'.
        forget_bias (float, optional): forget bias used when computing forget
            gate. It also can accept a boolean value `True`, which would set
            :math:`forget\\_bias` as 0 but initialize :math:`b_{f}` as 1 and
            :math:`b_{i}, b_{f}, b_{c}, b_{0}` as 0. This is recommended in
            http://www.jmlr.org/proceedings/papers/v37/jozefowicz15.pdf .
            Default 1.0.
        num_layers(int, optional): The number of LSTM to be stacked. Default 1.
        dropout(float|list|tuple, optional): The dropout probability after each
            LSTM. It also can be a list or tuple, including dropout probabilities
            for the corresponding LSTM. Default 0.0
        param_attr (list|tuple|ParamAttr): A list, tuple or something can be
            converted to a ParamAttr instance by `ParamAttr._to_attr`. If it is
            a list or tuple, it's length must equal to `num_layers`. Otherwise,
            construct a list by `StackedRNNCell.stack_param_attr(param_attr, num_layers)`.
            Default None.
        bias_attr (list|tuple|ParamAttr): A list, tuple or something can be
            converted to a ParamAttr instance by `ParamAttr._to_attr`. If it is
            a list or tuple, it's length must equal to `num_layers`. Otherwise,
            construct a list by `StackedRNNCell.stack_param_attr(bias_attr, num_layers)`.
            Default None.
        dtype(string, optional): The data type used in this cell. It can be
            float32 or float64. Default float32.

    Examples:

        .. code-block:: python

            import paddle
            import paddle.fluid as fluid
            from paddle.incubate.hapi.text import StackedLSTMCell, RNN

            inputs = paddle.rand((2, 4, 32))
            cell = StackedLSTMCell(input_size=32, hidden_size=64)
            rnn = RNN(cell=cell)
            outputs, _ = rnn(inputs)  # [2, 4, 64]
    """

    def __init__(self,
                 input_size,
                 hidden_size,
                 gate_activation=None,
                 activation=None,
                 forget_bias=1.0,
                 num_layers=1,
                 dropout=0.0,
                 param_attr=None,
                 bias_attr=None,
                 dtype="float32"):
        super(StackedLSTMCell, self).__init__()
        self.dropout = utils.convert_to_list(dropout, num_layers, "dropout",
                                             float)
        param_attrs = StackedRNNCell.stack_param_attr(param_attr, num_layers)
        bias_attrs = StackedRNNCell.stack_param_attr(bias_attr, num_layers)

        self.cells = []
        for i in range(num_layers):
            if forget_bias is True:
                bias_attrs[
                    i].initializer = fluid.initializer.NumpyArrayInitializer(
                        np.concatenate(
                            np.zeros(2 * hidden_size),
                            np.ones(hidden_size), np.zeros(hidden_size))
                        .astype(dtype))
                forget_bias = 0.0
            self.cells.append(
                self.add_sublayer(
                    "lstm_%d" % i,
                    BasicLSTMCell(
                        input_size=input_size if i == 0 else hidden_size,
                        hidden_size=hidden_size,
                        gate_activation=gate_activation,
                        activation=activation,
                        forget_bias=forget_bias,
                        param_attr=param_attrs[i],
                        bias_attr=bias_attrs[i],
                        dtype=dtype)))

    def forward(self, inputs, states):
        """
        Performs the stacked LSTM cells sequentially. Each cell's `inputs` is
        the `outputs` of the previous cell. And each cell's `states` is the
        corresponding one in `states`.

        Parameters:
            inputs (Variable): The inputs for the first cell. It is a float32 or
                float64 tensor with shape `[batch_size, input_size]`.
            states (list): A list containing states for all cells orderly.
            **kwargs: Additional keyword arguments, which passed to `cell.forward`
                for all including cells.

        Returns:
            tuple: A tuple( :code:`(outputs, new_states)` ), where `outputs` is \
                a tensor with shape `[batch_size, hidden_size]`, corresponding \
                to :math:`h_{t}` in the formula of the last LSTM; `new_states` \
                is a list composed of every LSTM `new_states` which is a pair \
                of tensors standing for :math:`h_{t}, c_{t}` in the formula, \
                and the data type and structure of these tensors all is same \
                as that of `states`.
        """
        new_states = []
        for i, cell in enumerate(self.cells):
            outputs, new_state = cell(inputs, states[i])
            outputs = layers.dropout(
                outputs,
                self.dropout[i],
                dropout_implementation='upscale_in_train') if self.dropout[
                    i] > 0 else outputs
            inputs = outputs
            new_states.append(new_state)
        return outputs, new_states

    @property
    def state_shape(self):
        """
        The `state_shape` of StackedLSTMCell is a list composed of each including
        LSTM cell's `state_shape`.

        Returns:
            list: A list composed of each including LSTM cell's `state_shape`.
        """
        return [cell.state_shape for cell in self.cells]


class LSTM(Layer):
    """
    Applies a stacked multi-layer long short-term memory (LSTM) RNN to an input
    sequence.

    The formula for LSTM used here is as follows:

    .. math::

        i_{t} & = act_g(W_{x_{i}}x_{t} + W_{h_{i}}h_{t-1} + b_{i})

        f_{t} & = act_g(W_{x_{f}}x_{t} + W_{h_{f}}h_{t-1} + b_{f} + forget\\_bias)

        c_{t} & = f_{t}c_{t-1} + i_{t} act_c (W_{x_{c}}x_{t} + W_{h_{c}}h_{t-1} + b_{c})

        o_{t} & = act_g(W_{x_{o}}x_{t} + W_{h_{o}}h_{t-1} + b_{o})

        h_{t} & = o_{t} act_c (c_{t})


    Parameters:
        input_size (int): The input feature size for the first LSTM.
        hidden_size (int): The hidden size for every LSTM.
        gate_activation (function, optional): The activation function for gates
            of LSTM, that is :math:`act_g` in the formula. Default: None,
            representing for `fluid.layers.sigmoid`.
        activation (function, optional): The non-gate activation function of
            LSTM, that is :math:`act_c` in the formula. Default: None,
            representing for 'fluid.layers.tanh'.
        forget_bias (float, optional): forget bias used when computing forget
            gate. It also can accept a boolean value `True`, which would set
            :math:`forget\\_bias` as 0 but initialize :math:`b_{f}` as 1 and
            :math:`b_{i}, b_{f}, b_{c}, b_{0}` as 0. This is recommended in
            http://www.jmlr.org/proceedings/papers/v37/jozefowicz15.pdf .
            Default 1.0.
        num_layers(int, optional): The number of LSTM to be stacked. Default 1.
        dropout(float|list|tuple, optional): The dropout probability after each
            LSTM. It also can be a list or tuple, including dropout probabilities
            for the corresponding LSTM. Default 0.0
        is_reverse (bool, optional): Indicate whether to calculate in the reverse
            order of input sequences. Default: `False`.
        time_major (bool, optional): Indicate the data layout of Tensor included
            in `input` and `output` tensors. If `False`, the data layout would
            be batch major with shape `[batch_size, sequence_length, ...]`.  If
            `True`, the data layout would be time major with shape
            `[sequence_length, batch_size, ...]`. Default: `False`.
        param_attr (list|tuple|ParamAttr): A list, tuple or something can be
            converted to a ParamAttr instance by `ParamAttr._to_attr`. If it is
            a list or tuple, it's length must equal to `num_layers`. Otherwise,
            construct a list by `StackedRNNCell.stack_param_attr(param_attr, num_layers)`.
            Default None.
        bias_attr (list|tuple|ParamAttr): A list, tuple or something can be
            converted to a ParamAttr instance by `ParamAttr._to_attr`. If it is
            a list or tuple, it's length must equal to `num_layers`. Otherwise,
            construct a list by `StackedRNNCell.stack_param_attr(bias_attr, num_layers)`.
            Default None.
        dtype(string, optional): The data type used in this cell. It can be
            float32 or float64. Default float32.

    Examples:

        .. code-block:: python

            import paddle
            import paddle.fluid as fluid
            from paddle.incubate.hapi.text import LSTM

            inputs = paddle.rand((2, 4, 32))
            lstm = LSTM(input_size=32, hidden_size=64, num_layers=2)
            outputs, _ = lstm(inputs)  # [2, 4, 64]
    """

    def __init__(self,
                 input_size,
                 hidden_size,
                 gate_activation=None,
                 activation=None,
                 forget_bias=1.0,
                 num_layers=1,
                 dropout=0.0,
                 is_reverse=False,
                 time_major=False,
                 param_attr=None,
                 bias_attr=None,
                 dtype='float32'):
        super(LSTM, self).__init__()
        lstm_cell = StackedLSTMCell(input_size, hidden_size, gate_activation,
                                    activation, forget_bias, num_layers,
                                    dropout, param_attr, bias_attr, dtype)
        self.lstm = RNN(lstm_cell, is_reverse, time_major)

    def forward(self, inputs, initial_states=None, sequence_length=None):
        """
        Performs the stacked multi-layer LSTM layer by layer. Each LSTM's `outputs`
        is the `inputs` of the subsequent one.

        Parameters:
            inputs (Variable): The inputs for the first LSTM. It is a float32
                or float64 tensor shaped `[batch_size, sequence_length, input_size]`.
            initial_states (list|None, optional): A list containing initial states 
                of all stacked LSTM, and the initial states of each LSTM is a pair
                of tensors shaped `[batch_size, hidden_size]`. If not provided,
                use 0 as initial states. Default None.
            sequence_length (Variable, optional): A tensor with shape `[batch_size]`.
                It stores real length of each instance, thus enables users to extract
                the last valid state when past a batch element's sequence length for
                correctness. If not provided, the paddings would be treated same as
                non-padding inputs. Default None.

        Returns:
            tuple: A tuple( :code:`(outputs, final_states)` ), where `outputs` \
                is the output of last LSTM and it is a tensor with shape \
                `[batch_size, sequence_length, hidden_size]` and has the same \
                data type as `inputs`, `final_states` is the counterpart of \
                `initial_states` at last time step, thus has the same structure \
                with it and has tensors with same shapes data types. 
        """
        return self.lstm(inputs, initial_states, sequence_length)
