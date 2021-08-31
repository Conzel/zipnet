{# 
    Template file for generating automated unit tests.
    Use the generate_tests.py script to regenerate.
#}
// This file has been automatically generated by Jinja2 via the 
// script {{ file }}.
// Please do not change this file by hand.
use ml::convolutions::*;
use ml::transposed_convolutions::*;
use ndarray::{Array, array, Dimension, Array4};

fn arr_allclose<D: Dimension>(arr1: &Array<f32,D>, arr2: &Array<f32,D>) -> bool {
    (arr1 - arr2).map(|x| (*x as f32).abs()).sum() < 1e-3
}

{% for t in random_tests %}
#[test]
fn test_py_implementation_random_arrays_{{t.test_name}}() {
    {% for r in t.random_test_objects %}
        let test_input{{loop.index}} = {{ r.input_arr }};
        {# The type hint is needed for the array-makro #}
        let kernel{{loop.index}}: Array4<f32> = {{ r.kernel }};
        let conv_layer{{loop.index}} = {{t.layer_name}}::new(kernel{{loop.index}}, {{r.stride}}, {{r.padding}});
        let target_output{{loop.index}} = {{ r.output_arr }};
        let current_output{{loop.index}} = conv_layer{{loop.index}}.{{t.function_name}}(&test_input{{loop.index}});

        assert!(arr_allclose(&current_output{{loop.index}}, &target_output{{loop.index}}), 
                "{:?} was not equal to {:?}", current_output{{loop.index}}, target_output{{loop.index}});
    {% endfor %}
}
{% endfor %}
