{
    "dataset_reader":{
        "type":"syntactic_then_semantic"
    },
    "train_data_path": "/Users/masashi-y/spigot/semeval2015_data/dm/data/english/english_dm_augmented_dev.sdp",
    "validation_data_path": "/Users/masashi-y/spigot/semeval2015_data/dm/data/english/english_dm_augmented_dev.sdp",
    "test_data_path": "/Users/masashi-y/spigot/semeval2015_data/dm/data/english/english_id_dm_augmented_test.sdp",
    "model": {
      "type": "syntactic_then_semantic",
      "syntactic_parser": {
        "type": "from_archive",
        #"archive_file": "/Users/masashi-y/allennlp/test_fixtures/basic_classifier/serialization/model.tar.gz"
        "archive_file": "https://allennlp.s3.amazonaws.com/models/biaffine-dependency-parser-ptb-2020.02.10.tar.gz"
      },
      "semantic_parser": {
        "type": "syntactically_informed_graph_parser"
      }
    },

    "data_loader": {
      "batch_sampler": {
        "type": "bucket",
        "batch_size": 128
      }
    },
    "trainer": {
      "num_epochs": 80,
      "grad_norm": 5.0,
      "patience": 50,
      "cuda_device": 1,
      "validation_metric": "+f1",
      "optimizer": {
        "type": "dense_sparse_adam",
        "betas": [0.9, 0.9]
      }
    }
  }

