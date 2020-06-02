{
  vocabulary: {
    type: 'from_files',
    directory: '/home/masashi-y/my_spigot/vocabulary',
  },
  dataset_reader: {
    type: 'syntactic_then_semantic',
  },
  validation_dataset_reader: {
    type: 'syntactic_then_semantic',
    skip_when_no_arcs: false
  },
  train_data_path: '/home/masashi-y/english/english_dm_augmented_train.sdp',
  validation_data_path: '/home/masashi-y/english/english_dm_augmented_dev.sdp',
  test_data_path: '/home/masashi-y/english/english_id_dm_augmented_test.sdp',
  model: {
    type: 'syntactic_then_semantic',
    share_text_field_embedder: true,
    share_pos_tag_embedding: true,
    decay_syntactic_loss: 0.9,
    freeze_syntactic_parser: false,
    edge_prediction_threshold: 0.5,
    syntactic_parser: {
      type: 'from_archive',
      archive_file: '/home/masashi-y/my_spigot/biaffine-dependency-parser-ptb-2020.02.10.fixed.tar.gz',
    },
    semantic_parser: {
      type: 'syntactically_informed_graph_parser',
      text_field_embedder: {
        token_embedders: {
          tokens: {
            type: 'embedding',
            embedding_dim: 100,
          },
        },
      },
      pos_tag_embedding: {
        embedding_dim: 100,
        vocab_namespace: 'pos',
        sparse: true,
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
    cuda_device: 1,
    validation_metric: '+f1',
    optimizer: {
      type: 'dense_sparse_adam',
      betas: [0.9, 0.9],
    },
  },
}
