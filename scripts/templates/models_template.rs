{# 
    Template file for generating the model specification. 
    Use the generate_models.py script to regenerate.
#}
//! This module provides the hierarchical models used in the encoding and decoding process.
// This file has been automatically generated by Jinja2 via the
// script {{ file }}.
// Please do not change this file by hand.
use crate::{
    activation_functions::{GdnLayer, IgdnLayer, ReluLayer},
    weight_loader::WeightLoader,
    WeightPrecision,
};
use convolutions_rs::{
    convolutions::ConvolutionLayer, transposed_convolutions::TransposedConvolutionLayer, Padding,
};
use ndarray::*;

pub type InternalDataRepresentation = Array3<WeightPrecision>;

// A note on the weights:
// Naming convention:
// [architecture]_[coder]_[layer type]_[layer]_[weight type]

/// General model trait for en- and decoding
pub trait CodingModel {
    fn forward_pass(&self, input: &InternalDataRepresentation) -> InternalDataRepresentation;
}

impl CodingModel for ConvolutionLayer<WeightPrecision> {
    fn forward_pass(&self, input: &InternalDataRepresentation) -> InternalDataRepresentation {
        self.convolve(input)
    }
}

impl CodingModel for TransposedConvolutionLayer<WeightPrecision> {
        fn forward_pass(&self, input: &InternalDataRepresentation) -> InternalDataRepresentation {
            self.transposed_convolve(input)
        }
}
    

impl CodingModel for GdnLayer {
    fn forward_pass(&self, input: &InternalDataRepresentation) -> InternalDataRepresentation {
        self.activate(input)
    }
}

impl CodingModel for IgdnLayer {
    fn forward_pass(&self, input: &InternalDataRepresentation) -> InternalDataRepresentation {
        self.activate(input)
    }
}

impl CodingModel for ReluLayer {
    fn forward_pass(&self, input: &InternalDataRepresentation) -> InternalDataRepresentation {
        self.activate(input)
    }
}

{% for m in models %} 
    pub struct {{m.rust_name}} {
        {% for l in m.layers %}
            {{l.python_name}}{{loop.index0}}: {{l.rust_name}}<WeightPrecision>,
            {% if l.activation is not none %}
            {{l.activation.python_name}}{{loop.index0}}: {{l.activation.rust_name}},
            {% endif %}
        {% endfor %}
    }

    impl CodingModel for {{m.rust_name}} {
        {# Have to allow since the last let might be extraneous due to model generation #}
        #[allow(clippy::let_and_return)]
        fn forward_pass(&self, input: &InternalDataRepresentation) -> InternalDataRepresentation {
            let x = input.clone();
            // trace!("input: {:?}\n", x);
            {% for l in m.layers %}
                let x = self.{{l.python_name}}{{loop.index0}}.forward_pass(&x);
                // trace!("{{l.python_name}}{{loop.index0}}_output: {:?}\n", x);
                {% if l.activation is not none %}
                    let x = self.{{l.activation.python_name}}{{loop.index0}}.forward_pass(&x);
                {% endif %}
                // trace!("{{l.activation.python_name}}{{loop.index0}}_output: {:?}\n", x);
            {% endfor %}
            x
        }
    }

    impl {{m.rust_name}} {
        pub fn new(loader: &mut impl WeightLoader) -> Self {
            {% for l in m.layers %}
                {% set weight_variable = l.python_name + loop.index0|string + "_weights" %}
                {% set weight_key = m.python_name + "." + l.python_name + loop.index0|string + ".weight.npy" %}
                let {{ weight_variable }} = loader.get_weight("{{weight_key}}",
                    ({{l.filters}}, {{l.channels}}, {{l.kernel_width}}, {{l.kernel_height}})
                ).unwrap();
                trace!("{{weight_key}}: {:?}\n", {{weight_variable}});
                let {{l.python_name}}{{loop.index0}} = {{l.rust_name}}::new({{weight_variable}}, {{l.stride}}, {{l.padding}});
                {% if l.activation is not none %}
                    let {{l.activation.python_name}}{{loop.index0}} = {{l.activation.rust_name}}::new();
                {% endif %}
            {% endfor %}
            Self {
                {% for l in m.layers %}
                    {{l.python_name}}{{loop.index0}},
                    {% if l.activation is not none %}
                    {{l.activation.python_name}}{{loop.index0}},
                    {% endif %}
                {% endfor %}
            }
        }
    }
{% endfor %}

mod tests {
    #[allow(unused_imports)]
    use crate::weight_loader::NpzWeightLoader;
    #[allow(unused_imports)]
    use super::*;

    {% for m in models %}
    #[test]
    fn smoke_test_{{m.rust_name.lower()}}() {
        let mut loader = NpzWeightLoader::full_loader();
        let _encoder = {{m.rust_name}}::new(&mut loader);
    }
    {% endfor %}
}
