from transformers import BartConfig


class MultiModalBartConfig(BartConfig):
    def __init__(
            self,
            activation_dropout=0.0,
            extra_pos_embeddings=2,
            activation_function="gelu",
            vocab_size=50320,
            image_feature_size=2048 + 4,
            d_model=1024,
            encoder_ffn_dim=4096,
            encoder_layers=12,
            encoder_attention_heads=16,
            decoder_ffn_dim=4096,
            decoder_layers=12,
            decoder_attention_heads=16,
            encoder_layerdrop=0.0,
            decoder_layerdrop=0.0,
            attention_dropout=0.0,
            dropout=0.1,
            max_position_embeddings=1024,
            init_std=0.02,
            classif_dropout=0.0,
            num_labels=1,
            num_attributes=1,
            num_relations=1,
            is_encoder_decoder=True,
            pad_token_id=1,
            bos_token_id=0,
            eos_token_id=2,
            img_feat_id=50273,
            cls_token_id=50276,
            normalize_before=False,
            add_final_layer_norm=False,
            scale_embedding=False,
            normalize_embedding=True,
            static_position_embeddings=False,
            add_bias_logits=False,
            decoder_start_token_id=0,
            partial_load=(),
            lm_loss_factor=1.0,
            mrm_loss_factor=1.0,
            attribute_loss_factor=1.0,
            relation_loss_factor=1.0,
            **common_kwargs
    ):
        super(MultiModalBartConfig, self).__init__(
            activation_dropout=activation_dropout,
            extra_pos_embeddings=extra_pos_embeddings,
            activation_function=activation_function,
            vocab_size=vocab_size,
            d_model=d_model,
            encoder_ffn_dim=encoder_ffn_dim,
            encoder_layers=encoder_layers,
            encoder_attention_heads=encoder_attention_heads,
            decoder_ffn_dim=decoder_ffn_dim,
            decoder_layers=decoder_layers,
            decoder_attention_heads=decoder_attention_heads,
            encoder_layerdrop=encoder_layerdrop,
            decoder_layerdrop=decoder_layerdrop,
            attention_dropout=attention_dropout,
            dropout=dropout,
            max_position_embeddings=max_position_embeddings,
            init_std=init_std,
            classif_dropout=classif_dropout,
            num_labels=num_labels,
            is_encoder_decoder=is_encoder_decoder,
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            normalize_before=normalize_before,
            add_final_layer_norm=add_final_layer_norm,
            scale_embedding=scale_embedding,
            normalize_embedding=normalize_embedding,
            static_position_embeddings=static_position_embeddings,
            add_bias_logits=add_bias_logits,
            decoder_start_token_id=decoder_start_token_id,
            **common_kwargs
        )

        self.image_feature_size = image_feature_size
        self.img_feat_id = img_feat_id
        self.cls_token_id = cls_token_id
        self.partial_load = partial_load
        self.num_attributes = num_attributes
        self.num_relations = num_relations
        self.lm_loss_factor = lm_loss_factor
        self.mrm_loss_factor = mrm_loss_factor
        self.attribute_loss_factor = attribute_loss_factor
        self.relation_loss_factor = relation_loss_factor
