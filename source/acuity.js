/* jshint esversion: 6 */
/* eslint "indent": [ "error", 4, { "SwitchCase": 1 } ] */

var acuity = acuity || {};
var json = json || require('./json');
var jsyaml = jsyaml || require('./jsyaml');
var python = python || require('./python');

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
                        return acuity.Data.open(context, context.identifier.replace('.json', '.data')).then((data) => {
                            return acuity.Quantization.open(context, context.identifier.replace('.json', '.quantize')).then((quantization) => {
                                return new acuity.Model(metadata, obj, data, quantization);
                            });
                        });
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

    constructor(metadata, model, modelData, quantization) {
        this._nodes = [];
        this._inputs = [];
        this._outputs = [];
        const args = new Map();
        const arg = (name) => {
            if (!args.has(name)) {
                args.set(name, { name: name, shape: null, quantization: null });
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

        if(quantization) {
            for (const key of Object.keys(quantization)) {
                const parameters = quantization[key];
                for (const tensorName of Object.keys(parameters)) {
                    if (args.has(tensorName)) {
                        const arg = args.get(tensorName);
                        arg.quantization = new acuity.Quantization(parameters[tensorName]);
                        arg.type.dataType = arg.quantization.dataType;
                    }
                }
            }
        }

        if (modelData) {
            for (const layerName of modelData.keys()) {
                for (const tensorName of modelData.get(layerName).keys()) {
                    const tensor = modelData.get(layerName).get(tensorName);
                    const tensorUrl = acuity.Utility.getTensorUrl(layerName, tensorName);
                    if (args.has(tensorUrl)) {
                        const arg = args.get(tensorUrl);
                        arg.type.shape.dimensions = tensor.shape;
                        arg.initializer = new acuity.Tensor(arg.type, arg.quantization, tensor.data);
                    }
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
                const tensorUrl = "@" + this._name + ":" + constant.name;
                const type = new acuity.TensorType(null, new acuity.TensorShape(null));
                const argument = new acuity.Argument(tensorUrl, type, null, new acuity.Tensor(type));
                this._inputs.push(new acuity.Parameter(constant.name, true, [ argument ]));
                args.set(tensorUrl, argument);
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
                    this._value += ' (default)';
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
        this._dataType = dataType || 'float32';
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

    constructor(type, quantization, data) {
        this._type = type;
        this._quantization = quantization || null;
        this._data = data || null;
    }

    get kind() {
        return 'Constant';
    }

    get type() {
        return this._type;
    }

    get state() {
        return this._context().state;
    }

    get value() {
        const context = this._context();
        if (context.state) {
            return null;
        }
        context.limit = Number.MAX_SAFE_INTEGER;
        return this._decode(context, 0);
    }

    toString() {
        const context = this._context();
        if (context.state) {
            return '';
        }
        context.limit = 10000;
        const value = this._decode(context, 0);
        return JSON.stringify(value, null, 4);
    }

    _context() {
        const context = {};
        context.state = null;
        context.index = 0;
        context.count = 0;

        if (this._data == null) {
            context.state = 'Tensor data is empty.';
            return context;
        }

        context.dataType = this._type.dataType;
        context.shape = this._type.shape.dimensions;
        context.data = new DataView(this._data.buffer, this._data.byteOffset, this._data.byteLength);

        if (this._quantization) {
            const elementCount = context.shape.reduce((a, c) => a * c);
            const dataTypeBytes = acuity.Utility.dataTypeBytes(this._quantization.dataType);
            this._quantizedData = new ArrayBuffer(elementCount * dataTypeBytes);
            const dataView = new DataView(this._quantizedData);
            for (let i = 0; i < elementCount; i++) {
                const floatValue = context.data.getFloat32(i * 4, true);
                const quantizedValue = this._quantization.quantize(floatValue);
                switch (this._quantization.dataType) {
                    case 'uint8':
                        dataView.setUint8(i * dataTypeBytes, quantizedValue, true);
                        break;
                    case 'int8':
                        dataView.setInt8(i * dataTypeBytes, quantizedValue, true);
                        break;
                    case 'uint16':
                        dataView.setUint16(i * dataTypeBytes, quantizedValue, true);
                        break;
                    case 'int16':
                        dataView.setInt16(i * dataTypeBytes, quantizedValue, true);
                        break;
                    case 'int32':
                        dataView.setInt32(i * dataTypeBytes, quantizedValue, true);
                        break;
                    case 'uint32':
                        dataView.setUint32(i * dataTypeBytes, quantizedValue, true);
                        break;
                    default:
                        break;
                }
            }
            context.data = dataView;
        }

        return context;
    }

    _decode(context, dimension) {
        const shape = (context.shape.length == 0) ? [ 1 ] : context.shape;
        const size = shape[dimension];
        const results = [];
        if (dimension == shape.length - 1) {
            for (let i = 0; i < size; i++) {
                if (context.count > context.limit) {
                    results.push('...');
                    return results;
                }
                switch (context.dataType) {
                    case 'uint8':
                        results.push(context.data.getUint8(context.index));
                        context.index += 1;
                        context.count++;
                        break;
                    case 'int8':
                        results.push(context.data.getInt8(context.index));
                        context.index += 1;
                        context.count++;
                        break;
                    case 'int16':
                        results.push(context.data.getInt16(context.index));
                        context.index += 2;
                        context.count++;
                        break;
                    case 'int32':
                        results.push(context.data.getInt32(context.index, true));
                        context.index += 4;
                        context.count++;
                        break;
                    case 'int64':
                        results.push(context.data.getInt64(context.index, true));
                        context.index += 8;
                        context.count++;
                        break;
                    case 'float16':
                        results.push(context.data.getFloat16(context.index, true));
                        context.index += 2;
                        context.count++;
                        break;
                    case 'float32':
                        results.push(context.data.getFloat32(context.index, true));
                        context.index += 4;
                        context.count++;
                        break;
                    case 'float64':
                        results.push(context.data.getFloat64(context.index, true));
                        context.index += 8;
                        context.count++;
                        break;
                    case 'string':
                        results.push(context.data[context.index++]);
                        context.count++;
                        break;
                    default:
                        break;
                }
            }
        }
        else {
            for (let j = 0; j < size; j++) {
                if (context.count > context.limit) {
                    results.push('...');
                    return results;
                }
                results.push(this._decode(context, dimension + 1));
            }
        }
        if (context.shape.length == 0) {
            return results[0];
        }
        return results;
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
            'elu', 'exp', 'floor', 'floor_div', 'hard_swish', 'leakyrelu', 'log', 'log_softmax',
            'neg', 'pow', 'prelu', 'quantize', 'relu', 'relu_keras', 'relun', 'rsqrt', 'sigmoid',
            'sin', 'softmax', 'softrelu', 'sqrt', 'square', 'tanh'
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

acuity.Data = class {
    static open(context, dataFileName) {
        return context.request(dataFileName).then((data) => {
            return context.require('./numpy').then((numpy) => {
                const array = new numpy.Array(data.peek());

                const unpickler = python.Unpickler.open(array.data);
                const execution = new python.Execution();
                const views = unpickler.load((name, args) => execution.invoke(name, args), null);

                acuity.Data._data = views.data[0];
                return acuity.Data._data;
            });

        }).catch(() => {
            acuity.Data._data = null;
            return acuity.Data._data;
        });
    }
};

acuity.Quantization = class {

    static open(context, dataFileName) {
        return context.request(dataFileName).then((data) => {
            return jsyaml.safeLoad(data.peek());
        }).catch(() => {
            return null;
        });
    }

    constructor(quantization) {
        this._quantization = quantization;
        this._doQuantize = function(quantization) {
            const range = acuity.Utility.rangeOfDataType(quantization.qtype);
            const limiter = function(value) {
                return Math.min(Math.max(value, range.min), range.max);
            };
            return function(value) {
                switch (quantization.dtype) {
                    case 'asymmetric_affine':
                        return limiter(Math.round(value / quantization.scale[0]) +
                            quantization.zero_point[0]);
                    default:
                        throw new acuity.Error("Unsupported qtype '" + quantization.dtype + "'");
                }
            };
        }(this._quantization);
    }

    get quantization() {
        return this._quantization;
    }

    get dataType() {
        return acuity.Utility.unifyDataType(this._quantization.qtype);
    }

    quantize(floatValue) {
        return this._doQuantize(floatValue);
    }

    toString() {
        if (this._quantization) {
            switch (this._quantization.dtype) {
                case "asymmetric_affine": {
                    let value = 'q';
                    const scale = (this._quantization.scale.length == 1) ? this._quantization.scale[0] : 0;
                    const zeroPoint = (this._quantization.zero_point.length == 1) ? this._quantization.zero_point[0] : 0;
                    if (scale != 0 || zeroPoint != 0) {
                        value = scale.toString() + ' * ' + (zeroPoint == 0 ? 'q' : ('(q - ' + zeroPoint.toString() + ')'));
                    }
                    if (this._quantization.min_value && this._quantization.min_value.length == 1) {
                        value = this._quantization.min_value[0].toString() + ' \u2264 ' + value;
                    }
                    if (this._quantization.max_value && this._quantization.max_value.length == 1) {
                        value = value + ' \u2264 ' + this._quantization.max_value[0].toString();
                    }
                    if (value != 'q') {
                        return value;
                    }
                    break;
                }
            }
        }

        return null;
    }
};

acuity.Utility = class {

    static getTensorUrl(layerName, port) {
        return "@" + layerName + ":" + port;
    }

    static parseTensorUrl(tensorUrl) {
        const parts = tensorUrl.split(':');
        const layerName = parts[0].split('@')[1];
        return { 'layerName': layerName, 'port': parts[1]};
    }

    static dataTypeBytes(dataType) {
        switch (acuity.Utility.unifyDataType(dataType)) {
            case 'uint8':
            case 'int8':
                return 1;
            case 'uint16':
            case 'int16':
            case 'float16':
                return 2;
            case 'uint32':
            case 'int32':
            case 'float32':
                return 4;
            case 'uint64':
            case 'int64':
            case 'float64':
                return 8;
            default:
                break;
        }
        return 0;
    }

    static unifyDataType(dataType) {
        const dataTypeMap = new Map([
            ['u8', 'uint8'], ['uint8', 'uint8'], ['i8', 'int8'], ['int8', 'int8'],
            ['u16', 'uint16'], ['u16', 'uint16'], ['i16', 'int16'], ['int16', 'int16'],
            ['u32', 'uint32'], ['u32', 'uint32'], ['i32', 'int32'], ['int32', 'int32'],
            ['u64', 'uint64'], ['u64', 'uint64'], ['i64', 'int64'], ['int64', 'int64'],
            ['f16', 'float16'], ['fp16', 'float16'],
            ['f32', 'float32'], ['fp32', 'float32'],
            ['f64', 'float64'], ['fp64', 'float64']
        ]);
        if (dataTypeMap.has(dataType)) {
            return dataTypeMap.get(dataType);
        }
        else {
            throw new acuity.Error("Unsupported dataType '" + dataType + "'");
        }
    }

    static rangeOfDataType(dataType) {
        const unifiedDataType = acuity.Utility.unifyDataType(dataType);
        const bits = 8 * acuity.Utility.dataTypeBytes(unifiedDataType);
        if (unifiedDataType.startsWith('u')) {
            return { min: 0, max: Math.pow(2, bits) - 1};
        }
        else {
            return { min: -Math.pow(2, bits - 1), max: Math.pow(2, bits - 1) - 1};
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