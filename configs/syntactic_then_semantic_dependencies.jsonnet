local pretrained_syntactic_parser = std.extVar('pretrained');

local syntactic_parser = if pretrained_syntactic_parser != '' then {
  type: 'from_archive',
  archive_file: pretrained_syntactic_parser,
} else {
  type: 'my_biaffine_parser',
  text_field_embedder: {
    token_embedders: {
      tokens: {
        type: 'embedding',
        embedding_dim: 100,
        pretrained_file: 'https://allennlp.s3.amazonaws.com/datasets/glove/glove.6B.100d.txt.gz',
        trainable: true,
      },
    },
  },
  pos_tag_embedding: {
    embedding_dim: 100,
    vocab_namespace: 'pos',
  },
  encoder: {
    type: 'stacked_bidirectional_lstm',
    input_size: 200,
    hidden_size: 400,
    num_layers: 3,
    recurrent_dropout_probability: 0.3,
    use_highway: true,
  },
  use_mst_decoding_for_validation: true,
  arc_representation_dim: 500,
  tag_representation_dim: 100,
  dropout: 0.3,
  input_dropout: 0.3,
  initializer: {
    regexes: [
      ['.*projection.*weight', { type: 'xavier_uniform' }],
      ['.*projection.*bias', { type: 'zero' }],
      ['.*tag_bilinear.*weight', { type: 'xavier_uniform' }],
      ['.*tag_bilinear.*bias', { type: 'zero' }],
      ['.*weight_ih.*', { type: 'xavier_uniform' }],
      ['.*weight_hh.*', { type: 'orthogonal' }],
      ['.*bias_ih.*', { type: 'zero' }],
      ['.*bias_hh.*', { type: 'lstm_hidden_bias' }],
    ],
  },
};

{
  vocabulary: {
    type: 'extend',
    directory: std.extVar('vocab'),
    only_include_pretrained_words: true,
  },
  dataset_reader: {
    type: 'syntactic_then_semantic',
  },
  validation_dataset_reader: {
    type: 'syntactic_then_semantic',
    skip_when_no_arcs: false,
  },
  train_data_path: std.extVar('train'),
  validation_data_path: std.extVar('dev'),
  test_data_path: std.extVar('test'),
  model: {
    type: 'syntactic_then_semantic',
    share_text_field_embedder: false,
    share_pos_tag_embedding: false,
    gumbel_sampling: true,
    // stop_syntactic_training_at_epoch: 40,
    decay_syntactic_loss: 1.0,
    freeze_syntactic_parser: false,
    edge_prediction_threshold: 0.5,
    syntactic_parser: syntactic_parser,
    semantic_parser: {
      type: 'syntactically_informed_graph_parser',
      text_field_embedder: {
        token_embedders: {
          tokens: {
            type: 'embedding',
            embedding_dim: 100,
            pretrained_file: 'https://allennlp.s3.amazonaws.com/datasets/glove/glove.6B.100d.txt.gz',
            sparse: false,
            trainable: true,
          },
        },
      },
      pos_tag_embedding: {
        embedding_dim: 100,
        vocab_namespace: 'pos',
        sparse: false,
      },
      encoder: {
        type: 'stacked_bidirectional_lstm',
        input_size: 200,
        hidden_size: 400,
        num_layers: 3,
        recurrent_dropout_probability: 0.3,
        use_highway: true,
      },
      arc_representation_dim: 500,
      tag_representation_dim: 100,
      dropout: 0.3,
      input_dropout: 0.3,
      initializer: {
        regexes: [
          ['.*feedforward.*weight', { type: 'xavier_uniform' }],
          ['.*feedforward.*bias', { type: 'zero' }],
          ['.*tag_bilinear.*weight', { type: 'xavier_uniform' }],
          ['.*tag_bilinear.*bias', { type: 'zero' }],
          ['.*weight_ih.*', { type: 'xavier_uniform' }],
          ['.*weight_hh.*', { type: 'orthogonal' }],
          ['.*bias_ih.*', { type: 'zero' }],
          ['.*bias_hh.*', { type: 'lstm_hidden_bias' }],
        ],
      },
    },
  },
  data_loader: {
    batch_sampler: {
      type: 'bucket',
      batch_size: 128,
    },
  },
  trainer: {
    num_epochs: 80,
    grad_clipping: 1.0,
    patience: 50,
    cuda_device: std.parseInt(std.extVar('device')),
    validation_metric: '+f1',
    optimizer: {
      type: 'adam',
      betas: [0.9, 0.9],
    },
  },
}
