import constriction
import numpy as np
import scipy.stats


def test_queue_gaussian():
    encoder = constriction.stream.queue.RangeEncoder()

    min_supported_symbol, max_supported_symbol = -100, 100
    symbols = np.array([23, -15, 78, 43, -69], dtype=np.int32)
    means = np.array([35.2, -1.7, 30.1, 71.2, -75.1], dtype=np.float64)
    stds = np.array([10.1, 25.3, 23.8, 35.4, 3.9], dtype=np.float64)

    encoder.encode_leaky_gaussian_symbols(
        symbols, min_supported_symbol, max_supported_symbol, means, stds)
    assert encoder.num_bits() == 64
    compressed = encoder.get_compressed()
    print(compressed)
    assert np.all(compressed == np.array(
        [473034731, 2276733146], dtype=np.uint32))

    decoder1 = constriction.stream.queue.RangeDecoder(compressed)
    reconstructed1 = decoder1.decode_leaky_gaussian_symbols(
        min_supported_symbol, max_supported_symbol, means, stds)
    assert decoder1.maybe_exhausted()
    assert np.all(reconstructed1 == symbols)

    decoder2 = encoder.get_decoder()
    reconstructed2 = decoder2.decode_leaky_gaussian_symbols(
        min_supported_symbol, max_supported_symbol, means, stds)
    assert decoder2.maybe_exhausted()
    assert np.all(reconstructed2 == symbols)


def test_stack_gaussian():
    encoder = constriction.stream.stack.AnsCoder()

    min_supported_symbol, max_supported_symbol = -100, 100
    symbols = np.array([23, -15, 78, 43, -69], dtype=np.int32)
    means = np.array([35.2, -1.7, 30.1, 71.2, -75.1], dtype=np.float64)
    stds = np.array([10.1, 25.3, 23.8, 35.4, 3.9], dtype=np.float64)

    encoder.encode_leaky_gaussian_symbols_reverse(
        symbols, min_supported_symbol, max_supported_symbol, means, stds)
    assert encoder.num_bits() == 64
    assert encoder.num_valid_bits() == 51
    compressed = encoder.get_compressed()
    assert np.all(compressed == np.array(
        [1109163715, 757457], dtype=np.uint32))

    decoder1 = constriction.stream.stack.AnsCoder(compressed)
    reconstructed1 = decoder1.decode_leaky_gaussian_symbols(
        min_supported_symbol, max_supported_symbol, means, stds)
    assert decoder1.is_empty()
    assert np.all(reconstructed1 == symbols)

    decoder2 = encoder
    reconstructed2 = decoder2.decode_leaky_gaussian_symbols(
        min_supported_symbol, max_supported_symbol, means, stds)
    assert decoder2.is_empty()
    assert np.all(reconstructed2 == symbols)


def test_chain_gaussian():
    rng = np.random.RandomState(123)
    original_data = rng.randint(2**32, size=100, dtype=np.uint32)
    decoder = constriction.stream.chain.ChainCoder(original_data, seal=True)

    min_supported_symbol, max_supported_symbol = -100, 100
    means = np.arange(50, dtype=np.float64)
    stds = np.array([10.0] * 50, dtype=np.float64)

    symbols = decoder.decode_leaky_gaussian_symbols(
        min_supported_symbol, max_supported_symbol, means, stds)

    remaining_prefix, remaining_suffix = decoder.get_remaining()
    print(len(remaining_prefix), len(remaining_suffix), len(original_data))
    assert len(remaining_prefix) + len(remaining_suffix) < len(original_data)

    # Variant 1: treat `remaining_prefix` and `remaining_suffix` separately
    encoder1 = constriction.stream.chain.ChainCoder(
        remaining_suffix, is_remaining=True)
    encoder1.encode_leaky_gaussian_symbols_reverse(
        symbols, min_supported_symbol, max_supported_symbol, means, stds)
    recovered_prefix1, recovered_suffix1 = encoder1.get_data(unseal=True)
    print(len(recovered_prefix1), len(recovered_suffix1), len(original_data))
    assert len(recovered_prefix1) == 0
    recovered1 = np.concatenate((remaining_prefix, recovered_suffix1))
    assert np.all(recovered1 == original_data)

    # Variant 2: concatenate `remaining_prefix` and `remaining_suffix`
    remaining = np.concatenate((remaining_prefix, remaining_suffix))
    encoder2 = constriction.stream.chain.ChainCoder(
        remaining, is_remaining=True)
    encoder2.encode_leaky_gaussian_symbols_reverse(
        symbols, min_supported_symbol, max_supported_symbol, means, stds)
    recovered_prefix2, recovered_suffix2 = encoder2.get_data(unseal=True)
    print(len(recovered_prefix2), len(recovered_suffix2), len(original_data))
    recovered2 = np.concatenate((recovered_prefix2, recovered_suffix2))
    assert np.all(recovered2 == original_data)

    # Variant 3: directly re-encode onto original coder
    encoder3 = decoder
    encoder3.encode_leaky_gaussian_symbols_reverse(
        symbols, min_supported_symbol, max_supported_symbol, means, stds)
    recovered_prefix3, recovered_suffix3 = encoder3.get_data(unseal=True)
    print(len(recovered_prefix3), len(recovered_suffix3), len(original_data))
    assert len(recovered_prefix3) == 0
    assert np.all(recovered_suffix3 == original_data)


def test_custom_model():
    symbols = np.array([3, 2, 6, -51, -19, 5, 87], dtype=np.int32)

    model_py = scipy.stats.norm(1.2, 4.9)
    model_iid = constriction.stream.model.CustomModel(
        model_py.cdf, model_py.ppf, -100, 100)

    model_parameters_iid = (
        np.array([[1.2, 4.9]], dtype=np.float64) +
        np.array([[0.0]]*len(symbols), dtype=np.float64)
    )
    model_parameters = np.array([[i, 4.9] for i in symbols], dtype=np.float64)
    model = constriction.stream.model.CustomModel(
        lambda x, params: scipy.stats.norm.cdf(
            x, loc=params[0], scale=params[1]),
        lambda x, params: scipy.stats.norm.ppf(
            x, loc=params[0], scale=params[1]),
        -100, 100)

    def test_coder(Encoder, Decoder, encode_iid, encode, expected_compressed_iid, expected_compressed):
        expected_compressed = np.array(expected_compressed, dtype=np.uint32)
        expected_compressed_iid = np.array(
            expected_compressed_iid, dtype=np.uint32)

        # Encode and decode i.i.d. symbols
        encoder = Encoder()
        encode_iid(encoder, symbols, model_iid)
        compressed = encoder.get_compressed()
        print(compressed)
        assert np.all(compressed == expected_compressed_iid)
        decoder = Decoder(compressed)
        reconstructed = decoder.decode_iid_custom_model(
            len(symbols), model_iid)
        assert np.all(reconstructed == symbols)

        # Encode and decode i.i.d. symbols, but with parameterized custom model.
        encoder = Encoder()
        encode(encoder, symbols, model, model_parameters_iid)
        compressed = encoder.get_compressed()
        print(compressed)
        assert np.all(compressed == expected_compressed_iid)
        decoder = Decoder(compressed)
        reconstructed = decoder.decode_custom_model(
            model, model_parameters_iid)
        assert np.all(reconstructed == symbols)

        # Encode and decode non-i.i.d. symbols.
        encoder = Encoder()
        encode(encoder, symbols, model, model_parameters)
        compressed = encoder.get_compressed()
        print(compressed)
        assert np.all(compressed == expected_compressed)
        decoder = Decoder(compressed)
        reconstructed = decoder.decode_custom_model(model, model_parameters)
        assert np.all(reconstructed == symbols)

    test_coder(
        constriction.stream.stack.AnsCoder,
        constriction.stream.stack.AnsCoder,
        lambda encoder, symbols, model: encoder.encode_iid_custom_model_reverse(
            symbols, model),
        lambda encoder, symbols, model, params: encoder.encode_custom_model_reverse(
            symbols, model, params),
        [3187671595, 2410106987,  48580], [3397926478, 6042])

    test_coder(
        constriction.stream.queue.RangeEncoder,
        constriction.stream.queue.RangeDecoder,
        lambda encoder, symbols, model: encoder.encode_iid_custom_model(
            symbols, model),
        lambda encoder, symbols, model, params: encoder.encode_custom_model(
            symbols, model, params),
        [2789142295, 3128556965, 414280666], [2147484271])


def test_huffman_queue():
    probabilities = np.array([0.3, 0.28, 0.12, 0.1, 0.2], dtype=np.float32)
    symbols = [1, 3, 2, 4, 0, 1, 4, 0, 2, 1]

    encoder = constriction.symbol.QueueEncoder()
    encoder_codebook = constriction.symbol.huffman.EncoderHuffmanTree(
        probabilities)
    for symbol in symbols:
        encoder.encode_symbol(symbol, encoder_codebook)
    compressed, compressed_len = encoder.get_compressed()
    print(compressed, compressed_len)
    assert compressed_len == 23
    assert np.all(compressed == np.array([3873993], dtype=np.uint32))

    decoder = encoder.get_decoder()
    decoder_codebook = constriction.symbol.huffman.DecoderHuffmanTree(
        probabilities)
    reconstructed = [decoder.decode_symbol(
        decoder_codebook) for _ in range(len(symbols))]
    assert reconstructed == symbols
