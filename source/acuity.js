/* jshint esversion: 6 */
/* eslint "indent": [ "error", 4, { "SwitchCase": 1 } ] */

var acuity = acuity || {};
var json = json || require('./json');

acuity.ModelFactory = class {

    match(context) {
        const extension = context.identifier.split('.').pop().toLowerCase();
        if (extension === 'json') {
            const obj = context.open('json');
            if (obj && obj.MetaData && obj.Layers) {
                return true;
            }
        }
        return false;
    }

    open(context) {
        return acuity.Metadata.open(context).then((metadata) => {
            const extension = context.identifier.split('.').pop().toLowerCase();
            switch (extension) {
                case 'json': {
                    const obj = context.open('json');
                    if (obj && obj.MetaData && obj.Layers) {
                        return new acuity.Model(metadata, obj);
                    }
                }
            }
        });
    }
};

acuity.Model = class {

    constructor(metadata, model, data, quantization) {
        this._name = model.MetaData.Name;
        this._format = 'Acuity ' + 'v' + model.MetaData.AcuityVersion;
        this._runtime = model.MetaData.Platform;
        this._graphs = [ new acuity.Graph(metadata, model, data, quantization) ];
    }

    get format() {
        return this._format;
    }

    get name() {
        return this._name;
    }

    get runtime() {
        return this._runtime;
    }

    get graphs() {
        return this._graphs;
    }
};

acuity.Graph = class {

    constructor(metadata, model) {
        this._nodes = [];
        this._inputs = [];
        this._outputs = [];
        const args = new Map();
        const arg = (name) => {
            if (!args.has(name)) {
                args.set(name, { name: name, shape: null });
            }
            return args.get(name);
        };

        for (const layerName of Object.keys(model.Layers)) {
            const layer = model.Layers[layerName];
            layer.inputs = layer.inputs.map((input) => {
                return arg(input);
            });
            layer.outputs = layer.outputs.map((port) => {
                const argument = arg("@" + layerName + ":" + port);
                let shape = null;
                if (layer.op.toLowerCase() == 'input' ||
                    layer.op.toLowerCase() == 'variable') {
                    if (Object.prototype.hasOwnProperty.call(layer.parameters, 'shape') && layer.parameters.shape.length > 0) {
                        shape = layer.parameters.shape;
                    }
                    else if (Object.prototype.hasOwnProperty.call(layer.parameters, 'size') && Object.prototype.hasOwnProperty.call(layer.parameters, 'channels')) {
                        const sizes = layer.parameters.size.split(' ');
                        shape = [0, parseInt(sizes[0]), parseInt(sizes[1]), layer.parameters.channels];
                    }
                    if(shape) {
                        shape = shape.map(x => x == 0 ? 1 : x);
                    }
                }
                argument.shape = shape;
                return argument;
            });
        }

        new acuity.Inference(model.Layers);

        for (const pair of args) {
            const type = new acuity.TensorType(null, new acuity.TensorShape(pair[1].shape));
            const arg = new acuity.Argument(pair[0], type, null, null);
            args.set(pair[0], arg);
        }

        for (const layerName of Object.keys(model.Layers)) {
            const layer = model.Layers[layerName];
            switch (layer.op.toLowerCase()) {
                case 'input': {
                    this._inputs.push(new acuity.Parameter(layerName, true, [
                        args.get(layer.outputs[0].name)
                    ]));
                    break;
                }
                case 'output': {
                    this._outputs.push(new acuity.Parameter(layerName, true, [
                        args.get(layer.inputs[0].name)
                    ]));
                    break;
                }
                default: {
                    this._nodes.push(new acuity.Node(metadata, layerName, layer, args));
                    break;
                }
            }
        }
    }

    get inputs() {
        return this._inputs;
    }

    get outputs() {
        return this._outputs;
    }

    get nodes() {
        return this._nodes;
    }
};

acuity.Node = class {

    constructor(metadata, name, layer, args) {
        this._metadata = metadata;
        this._name = name;
        this._type = layer.op;
        this._inputs = [];
        this._outputs = [];
        this._attributes = [];
        this._layer = layer;

        const schema = this._metadata.type(layer.op);
        if (schema) {
            if (layer.parameters) {
                for (const key of Object.keys(layer.parameters)) {
                    const metadata = this._metadata.attribute(this._type, key);
                    this._attributes.push(new acuity.Attribute(metadata, key, layer.parameters[key]));
                }
            }
        }

        for (let i = 0; i < layer.inputs.length; i++) {
            const input = layer.inputs[i];
            const arg = args.get(input.name);
            const name = schema && schema.inputs && i < schema.inputs.length ? schema.inputs[i].name : 'input' + i.toString();
            this._inputs.push(new acuity.Parameter(name, true, [ arg ]));
        }

        if (schema && schema.constants) {
            for (const constant of schema.constants) {
                // const name = "@" + this._name + ":" + constant.name;
                const type = new acuity.TensorType(null, new acuity.TensorShape(null));
                const argument = new acuity.Argument('', type, null, new acuity.Tensor(type));
                this._inputs.push(new acuity.Parameter(constant.name, true, [ argument ]));
            }
        }

        for (let i = 0; i < layer.outputs.length; i++) {
            const output = layer.outputs[i];
            const arg = args.get(output.name);
            const name = schema && schema.outputs && i < schema.outputs.length ? schema.outputs[i].name : 'output' + i.toString();
            this._outputs.push(new acuity.Parameter(name, true, [arg]));
        }
    }

    get type() {
        return this._type;
    }

    get name() {
        return this._name;
    }

    get metadata() {
        return this._metadata.type(this.type);
    }

    get inputs() {
        return this._inputs;
    }

    get outputs() {
        return this._outputs;
    }

    get attributes() {
        return this._attributes;
    }
};

acuity.Attribute = class {

    constructor(metadata, name, value) {
        this._type = null;
        this._name = name;
        this._value = value;
        if (metadata) {
            this._type = metadata.type || null;
            if (Object.prototype.hasOwnProperty.call(metadata, 'default')) {
                if (metadata.default === value) {
                    this._visible = false;
                }
            }
        }
    }

    get name() {
        return this._name;
    }

    get type() {
        return this._type;
    }

    get value() {
        return this._value;
    }

    get visible() {
        return this._visible == false ? false : true;
    }
};

acuity.Parameter = class {

    constructor(name, visible, args) {
        this._name = name;
        this._visible = visible;
        this._arguments = args;
        if (this._arguments.some((arg) => !arg)) {
            throw "";
        }
    }

    get name() {
        return this._name;
    }

    get visible() {
        return this._visible;
    }

    get arguments() {
        return this._arguments;
    }
};

acuity.Argument = class {

    constructor(name, type, quantization, initializer) {
        if (typeof name !== 'string') {
            throw new acuity.Error("Invalid argument identifier '" + JSON.stringify(name) + "'.");
        }
        this._name = name;
        this._type = type || null;
        this._quantization = quantization || null;
        this._initializer = initializer || null;
    }

    get name() {
        return this._name;
    }

    get type() {
        return this._type;
    }

    get quantization() {
        return this._quantization;
    }

    set quantization(quantization) {
        this._quantization = quantization;
    }

    get initializer() {
        return this._initializer;
    }

    set initializer(initializer) {
        this._initializer = initializer;
    }
};

acuity.TensorType = class {

    constructor(dataType, shape) {
        this._dataType = dataType || '?';
        this._shape = shape;
    }

    get dataType() {
        return this._dataType;
    }

    set dataType(dataType) {
        this._dataType = dataType;
    }

    get shape() {
        return this._shape;
    }

    set shape(shape) {
        this._shape = shape;
    }

    toString() {
        return (this.dataType || '?') + this._shape.toString();
    }
};

acuity.TensorShape = class {

    constructor(dimensions) {
        this._dimensions = dimensions || null;
    }

    get dimensions() {
        return this._dimensions;
    }

    set dimensions(dimensions) {
        this._dimensions = dimensions;
    }

    toString() {
        if (!this._dimensions || this._dimensions.length == 0) {
            return '';
        }
        return '[' + this._dimensions.map((dimension) => dimension.toString()).join(',') + ']';
    }
};

acuity.Tensor = class {

    constructor(type) {
        this._type = type;
    }

    get kind() {
        return 'Constant';
    }

    get type() {
        return this._type;
    }

    get state() {
        return 'Not supported.';
    }

    toString() {
        return '';
    }
};

acuity.Metadata = class {

    static open(context) {
        if (acuity.Metadata._metadata) {
            return Promise.resolve(acuity.Metadata._metadata);
        }
        return context.request('acuity-metadata.json', 'utf-8', null).then((data) => {
            acuity.Metadata._metadata = new acuity.Metadata(data);
            return acuity.Metadata._metadata;
        }).catch(() => {
            acuity.Metadata._metadata = new acuity.Metadata(null);
            return acuity.Metadata._metadata;
        });
    }

    constructor(data) {
        this._map = new Map();
        if (data) {
            const metadata = JSON.parse(data);
            this._map = new Map(metadata.map((item) => [ item.name, item ]));
        }
    }

    type(name) {
        return this._map.get(name);
    }

    attribute(type, name) {
        const schema = this.type(type);
        if (schema) {
            let attributeMap = schema.attributeMap;
            if (!attributeMap) {
                attributeMap = {};
                if (schema.attributes) {
                    for (const attribute of schema.attributes) {
                        attributeMap[attribute.name] = attribute;
                    }
                }
                schema.attributeMap = attributeMap;
            }
            const attributeSchema = attributeMap[name];
            if (attributeSchema) {
                return attributeSchema;
            }
        }
        return null;
    }
};

acuity.Inference =  class {

    constructor(layers) {
        this._outputs = new Map();
        const outputLayers = [];
        for (const layerName of Object.keys(layers)) {
            const layer = layers[layerName];
            if (layer.op.toLowerCase() == 'output') {
                outputLayers.push(layer);
            }
            for (const output of layer.outputs) {
                this._outputs.set(output.name, layer);
            }
        }
        this._passthroughs = new Set([
            'a_times_b_plus_c', 'abs', 'cast', 'clipbyvalue', 'dequantize', 'dtype_converter',
            'elu', 'exp', 'floor', 'hard_swish', 'leakyrelu', 'log', 'log_softmax',
            'neg', 'pow', 'prelu', 'quantize', 'relu', 'relu_keras', 'relun', 'rsqrt', 'sigmoid',
            'sin', 'softmax', 'softrelu', 'sqrt', 'square', 'tanh', 'layernormalize', 'batchnormalize',
            'reverse', 'cast', 'mish', 'round'
        ]);
        this._broadcats = new Set([
            'add', 'greater', 'greater_equal', 'less', 'less_equal', 'equal', 'not_equal',
            'floor_div', 'logical_or', 'logical_and', 'minimum', 'pow', 'real_div', 'squared_difference',
            'fllor_mod', 'subtract', 'multiply'
        ]);
        this._operators = new Map();
        this._operators.set('concat', (inputs, parameters) => {
            const outputShape = inputs[0].slice();
            outputShape[parameters.dim] = 0;
            for (const shape of inputs) {
                outputShape[parameters.dim] += shape[parameters.dim];
            }
            return [outputShape];
        });
        this._operators.set('convolution', (inputs, parameters) => {
            if (parameters.padding == 'VALID') {
                const out_h = ~~((inputs[0][1] + parameters.stride_h - parameters.ksize_h) / parameters.stride_h);
                const out_w = ~~((inputs[0][2] + parameters.stride_w - parameters.ksize_w) / parameters.stride_w);
                return [[inputs[0][0], out_h, out_w, parameters.weights]];
            }
            else if (parameters.padding == 'SAME') {
                const out_h = ~~((inputs[0][1] + parameters.stride_h - 1) / parameters.stride_h);
                const out_w = ~~((inputs[0][2] + parameters.stride_w - 1) / parameters.stride_w);
                return [[inputs[0][0], out_h, out_w, parameters.weights]];
            }
        });
        this._operators.set('fullconnect', (inputs, parameters) => {
            return [inputs[0].slice(0, parameters.axis).concat([parameters.weights])];
        });
        this._operators.set('pooling', (inputs, parameters) => {
            if (parameters.padding == 'VALID') {
                const out_h = ~~((inputs[0][1] + parameters.stride_h - parameters.ksize_h) / parameters.stride_h);
                const out_w = ~~((inputs[0][2] + parameters.stride_w - parameters.ksize_w) / parameters.stride_w);
                return [[inputs[0][0], out_h, out_w, inputs[0][3]]];
            }
            else if (parameters.padding == 'SAME') {
                const out_h = ~~((inputs[0][1] + parameters.stride_h - 1) / parameters.stride_h);
                const out_w = ~~((inputs[0][2] + parameters.stride_w - 1) / parameters.stride_w);
                return [[inputs[0][0], out_h, out_w, inputs[0][3]]];
            }
        });
        this._operators.set('reshape', (inputs, parameters) => {
            const negativeIndexs = [];
            const newShape = parameters.shape.map((item, index) => {
                if(item == 0) {
                    return inputs[0][index];
                }
                else if(item == -1) {
                    negativeIndexs.push(index);
                    return 1;
                }
                else {
                    return item;
                }
            });
            if(negativeIndexs.length > 0) {
                newShape[negativeIndexs[0]] = inputs[0].reduce((a, c) => a * c) / newShape.reduce((a, c) => a * c);
            }
            return [newShape];
        });
        this._operators.set('permute', (inputs, parameters) => {
            const newShape = inputs[0].map((item, index) => {
                return inputs[0][parameters.perm[index]];
            });
            return [newShape];
        });
        this._operators.set('matmul', (inputs, parameters) => {
            const a = inputs[0];
            const b = inputs[1];
            let newShape = a.slice(0, -2);
            if(parameters.transpose_a) {
                newShape = newShape.concat(a.slice(-1));
            }
            else {
                newShape = newShape.concat(a.slice(-2, -1));
            }
            if(parameters.transpose_b) {
                newShape = newShape.concat(b.slice(-2, -1));
            }
            else {
                newShape = newShape.concat(b.slice(-1));
            }
            return [newShape];
        });
        this._operators.set('stridedslice', (inputs, parameters) => {
            const input_shape = inputs[0].slice();
            const begin = parameters.slice_begin.slice();
            const end = parameters.slice_end.slice();
            if(parameters.slice_begin_mask > 0) {
                for(let i = 0; i < begin.length; i++) {
                    if((parameters.slice_begin_mask >>> i) & 0x1) {
                        begin[i] = -1;
                    }
                }
            }
            if(parameters.slice_end_mask > 0) {
                for(let i = 0; i < end.length; i++) {
                    if((parameters.slice_end_mask >>> i) & 0x1) {
                        end[i] = -1;
                    }
                }
            }
            for(let i = 0; i < begin.length; i++) {
                if(begin[i] == -1) {
                    begin[i] = 0;
                }
            }
            if(inputs[0].length == end.length){
                for(let i = 0; i < end.length; i++) {
                    if(end[i] == -1 || end[i] > input_shape[i]) {
                        end[i] = input_shape[i];
                    }
                }
            }
            else if(inputs[0].length < end.length){
                if(parameters.slice_new_axis_mask) {
                    const len = (parameters.slice_new_axis_mask >>> 0).toString(2).length;
                    for(let i = 0; i < len; i++) {
                        if((parameters.slice_new_axis_mask >>> i) & 0x1) {
                            input_shape.splice(i, 0, 1);
                        }
                    }
                    for(let i = 0; i < end.length; i++) {
                        if(end[i] == -1) {
                            end[i] = input_shape[i];
                        }
                    }
                }
            }

            let newShape = [];
            for(let i = 0; i < begin.length; i++) {
                newShape = newShape.concat([(end[i] - begin[i])/parameters.slice_strides[i]]);
            }
            if(parameters.slice_new_axis_mask) {
                const len = (parameters.slice_new_axis_mask >>> 0).toString(2).length;
                for(let i = 0; i < len; i++) {
                    if((parameters.slice_new_axis_mask >>> i) & 0x1) {
                        if(inputs[0].length == begin.length) {
                            newShape.splice(i, 0, 1);
                        }
                        else if(inputs[0].length < begin.length) {
                            newShape[i] = 1;
                        }
                    }
                }
            }
            return [newShape];
        });
        this._operators.set('broadcast', (inputs) => {
            const a = inputs[0];
            const b = inputs[1];
            const longer = a.length >= b.length ? a.slice() : b.slice();
            const shorter = a.length < b.length ? a.slice() : b.slice();
            const remain = longer.length - shorter.length;
            for(let i = 0; i < remain; i++) {
                shorter.splice(0, 0, 1);
            }
            for(let i = 0; i < longer.length; i++) {
                longer[i] = longer[i] > shorter[i] ? longer[i] : shorter[i];
            }
            return [longer];
        });
        this._operators.set('pad', (inputs, parameters) => {
            const newShape = inputs[0].map((item, index) => {
                return item + parameters.padding_value[index][0] + parameters.padding_value[index][1];
            });
            return [newShape];
        });
        this._operators.set('space2depth', (inputs, parameters) => {
            const h = inputs[0][1] / parameters.block_size[0];
            const w = inputs[0][2] / parameters.block_size[1];
            const c = inputs[0][3] * parameters.block_size[1] * parameters.block_size[1];
            return [[inputs[0][0], h, w, c]];
        });
        this._operators.set('slice', (inputs, parameters) => {
            const newShape = parameters.size.map((item, index) => {
                return item == -1 ? inputs[0][index] : item;
            });
            return [newShape];
        });
        this._operators.set('lstm', (inputs, parameters) => {
            let batch = inputs[0][0];
            const output = parameters.num_proj != null ? parameters.num_proj : parameters.weights;
            if(parameters.time_major) {
                batch = inputs[0][1];
            }
            let newShape = [];
            if(parameters.return_sequences) {
                newShape = [inputs[0][0], inputs[0][1], output];
            }
            else {
                newShape = [batch, output];
            }
            return [newShape, [batch, output], [batch, parameters.weights]];
        });
        this._operators.set('deconvolution', (inputs, parameters) => {
            const newShape = parameters.output_shape.map((item, index) => {
                return item == 0 ? inputs[0][index] : item;
            });
            return [newShape];
        });
        this._operators.set('gather', (inputs, parameters) => {
            const prefix = inputs[1].slice();
            const suffix = inputs[0].slice(parameters.axis + 1);
            const newShape = prefix.concat(suffix);
            return [newShape];
        });
        this._operators.set('conv1d', (inputs, parameters) => {
            if (parameters.padding == 'VALID') {
                const out_h = ~~((inputs[0][1] + parameters.stride - parameters.ksize) / parameters.stride);
                return [[inputs[0][0], out_h, parameters.weights]];
            }
            else if (parameters.padding == 'SAME') {
                const out_h = ~~((inputs[0][1] + parameters.stride - 1) / parameters.stride);
                return [[inputs[0][0], out_h, parameters.weights]];
            }
        });
        for (const layer of outputLayers) {
            for (const output of layer.outputs) {
                this._infer(output);
            }
        }
    }

    _infer(output) {
        if (this._outputs.has(output.name)) {
            let inputShapeReady = true;
            const layer = this._outputs.get(output.name);
            for (const input of layer.inputs) {
                if (input.shape === null) {
                    this._infer(input);
                    if (input.shape === null) {
                        inputShapeReady = false;
                        break;
                    }
                }
            }

            if (inputShapeReady) {
                let callback = null;
                if (this._operators.has(layer.op)) {
                    callback = this._operators.get(layer.op);
                }
                else if (this._passthroughs.has(layer.op)) {
                    callback = (inputs) => [ inputs[0].slice() ];
                }
                else if (this._broadcats.has(layer.op)) {
                    callback = this._operators.get('broadcast');
                }
                else {
                    callback = () => [];
                }
                const parameters = layer.parameters;
                const inputs = layer.inputs.map((input) => input.shape);
                const outputs = callback(inputs, parameters);
                for (let i = 0; i < outputs.length; i++) {
                    if (i < layer.outputs.length) {
                        layer.outputs[i].shape = outputs[i];
                    }
                }
            }
        }
    }
};

acuity.Error = class extends Error {

    constructor(message) {
        super(message);
        this.name = 'Error loading Acuity model.';
    }
};

if (typeof module !== 'undefined' && typeof module.exports === 'object') {
    module.exports.ModelFactory = acuity.ModelFactory;
}