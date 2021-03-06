#!/usr/bin/env python
"""
    Training on a single process
"""
from __future__ import division

import argparse
import os
import random
import torch
import torch.nn as nn
from torch.nn.init import xavier_uniform_

import onmt.opts as opts

from collections import defaultdict, OrderedDict

from onmt.inputters.inputter import (build_dataset_iter, lazily_load_dataset,
    _load_fields, _collect_report_features)
from onmt.model_builder import (build_model, build_embeddings_then_encoder,
    build_decoder_and_generator)
from onmt.utils.optimizers import build_optim
from onmt.trainer import build_trainer
from onmt.models import build_model_saver
from onmt.utils.logging import init_logger, logger


def _check_save_model_path(opt):
    save_model_path = os.path.abspath(opt.save_model)
    model_dirname = os.path.dirname(save_model_path)
    if not os.path.exists(model_dirname):
        os.makedirs(model_dirname)


def _tally_parameters(model):
    n_params = sum([p.nelement() for p in model.parameters()])
    enc = 0
    dec = 0
    for name, param in model.named_parameters():
        if 'encoder' in name:
            enc += param.nelement()
        elif 'decoder' or 'generator' in name:
            dec += param.nelement()
    return n_params, enc, dec


def training_opt_postprocessing(opt):
    if opt.word_vec_size != -1:
        opt.src_word_vec_size = opt.word_vec_size
        opt.tgt_word_vec_size = opt.word_vec_size

    if opt.layers != -1:
        opt.enc_layers = opt.layers
        opt.dec_layers = opt.layers

    opt.brnn = (opt.encoder_type == "brnn")

    if opt.rnn_type == "SRU" and not opt.gpuid:
        raise AssertionError("Using SRU requires -gpuid set.")

    if torch.cuda.is_available() and not opt.gpuid:
        logger.info("WARNING: You have a CUDA device, should run with -gpuid")

    if opt.seed > 0:
        torch.manual_seed(opt.seed)
        # this one is needed for torchtext random call (shuffled iterator)
        # in multi gpu it ensures datasets are read in the same order
        random.seed(opt.seed)
        # some cudnn methods can be random even after fixing the seed
        # unless you tell it to be deterministic
        torch.backends.cudnn.deterministic = True

    if opt.gpuid:
        torch.cuda.set_device(opt.device_id)
        if opt.seed > 0:
            # These ensure same initialization in multi gpu mode
            torch.cuda.manual_seed(opt.seed)

    return opt


def build_data_iter_fct(dataset_name, path_, fields_, opt_):

    def train_iter_wrapper():
        return build_dataset_iter(lazily_load_dataset(dataset_name, path_),
                                  fields_,
                                  opt_)

    return train_iter_wrapper


def main(opt):
    opt = training_opt_postprocessing(opt)
    init_logger(opt.log_file)
    # Load checkpoint if we resume from a previous training.
    if opt.train_from:
        logger.info('Loading checkpoint from %s' % opt.train_from)
        checkpoint = torch.load(opt.train_from,
                                map_location=lambda storage, loc: storage)
        model_opt = checkpoint['opt']
    else:
        checkpoint = None
        model_opt = opt

    # For each dataset, load fields generated from preprocess phase.
    train_iter_fcts = OrderedDict()
    valid_iter_fcts = OrderedDict()

    # Loop to create encoders, decoders, and generators
    encoders = OrderedDict()
    decoders = OrderedDict()

    generators = OrderedDict()
    src_vocabs = OrderedDict()
    tgt_vocabs = OrderedDict()

    # variables needed for sharing the same embedding matrix across encoders and decoders
    firstTime=True
    weightToShare=None

    # we share the word embedding space when source lang and target lang are the same
    mapLang2Emb = {}

    for (src_tgt_lang), data_path in zip(opt.src_tgt, opt.data):
        src_lang, tgt_lang = src_tgt_lang.split('-')
        # Peek the first dataset to determine the data_type.
        # (All datasets have the same data_type).
        # TODO: change this part to support different data type
        first_dataset = next(lazily_load_dataset("train", data_path))
        data_type = first_dataset.data_type
        fields = _load_fields(first_dataset, data_type, data_path, checkpoint)

        # Report src/tgt features.
        src_features, tgt_features = _collect_report_features(fields)
        for j, feat in enumerate(src_features):
            logger.info(' * src feature %d size = %d'
                        % (j, len(fields[feat].vocab)))
        for j, feat in enumerate(tgt_features):
            logger.info(' * tgt feature %d size = %d'
                        % (j, len(fields[feat].vocab)))

        # Build model.
        encoder, src_embeddings = build_embeddings_then_encoder(model_opt, fields)

        encoders[src_lang] = encoder

        decoder, generator, tgt_embeddings = build_decoder_and_generator(model_opt, fields)

        decoders[tgt_lang] = decoder
	
        # Share the embedding matrix across all the encoders and decoders - preprocess with share_vocab required.
        if model_opt.share_embeddings and firstTime:
                tgt_embeddings.word_lut.weight = src_embeddings.word_lut.weight
                weightToShare = src_embeddings.word_lut.weight
        if model_opt.share_embeddings and (not firstTime):
                tgt_embeddings.word_lut.weight = weightToShare
                src_embeddings.word_lut.weight = weightToShare
        firstTime = False

        if src_lang in mapLang2Emb:
                encoder.embeddings.word_lut.weight = mapLang2Emb.get(src_lang)
        else:
                mapLang2Emb[src_lang] = src_embeddings.word_lut.weight
        if tgt_lang in mapLang2Emb:
                decoder.embeddings.word_lut.weight = mapLang2Emb.get(tgt_lang)
        else:
                mapLang2Emb[tgt_lang] = tgt_embeddings.word_lut.weight 

        # TODO: fields from vocab?
        src_vocabs[src_lang] = fields['src'].vocab
        tgt_vocabs[tgt_lang] = fields['tgt'].vocab

        generators[tgt_lang] = generator

        # add this dataset iterator to the training iterators
        train_iter_fcts[(src_lang, tgt_lang)] = build_data_iter_fct('train',
                                                                    data_path,
                                                                    fields,
                                                                    opt)
        # add this dataset iterator to the validation iterators
        valid_iter_fcts[(src_lang, tgt_lang)] = build_data_iter_fct('valid',
                                                                    data_path,
                                                                    fields,
                                                                    opt)

    # build the model with all of the encoders and all of the decoders
    # note here we just replace the encoders of the final model
    # WORKING: move all of this to a function
    # TODO: don't call build model at all, new model creation logic here
    # TODO: note that build_base_model is also used when _restarting_ training
    # TODO: assert that losses, etc... get built correctly when restoring from
    # TODO:   checkpoint

    # NOTE: checkpoint currently hard-coded to None because we set it below
    model = build_model(model_opt, opt, fields, checkpoint=None)

    # TODO: this is a hack -- move to actually initializing the model with
    # TODO:   encoders and decoders
    encoder_ids = {lang_code: idx
                   for lang_code, idx
                   in zip(encoders.keys(), range(len(list(encoders.keys()))))}
    encoders = nn.ModuleList(encoders.values())
    model.encoder_ids = encoder_ids
    model.encoders = encoders

    decoder_ids = {lang_code: idx
                   for lang_code, idx
                   in zip(decoders.keys(), range(len(list(decoders.keys()))))}
    decoders = nn.ModuleList(decoders.values())
    model.decoder_ids = decoder_ids
    model.decoders = decoders

    # we attach these to the model for persistence
    model.src_vocabs = src_vocabs
    model.tgt_vocabs = tgt_vocabs

    model.generators = nn.ModuleList(generators.values())

    if len(opt.gpuid) > 0:
        model.to('cuda')

    # Load the model states from checkpoint or initialize them.
    if checkpoint is not None:
        model.load_state_dict(checkpoint['model'])
        #generator.load_state_dict(checkpoint['generator'])
    else:
        logger.info("Initializing model parameters")
        if model_opt.param_init != 0.0:
            for p in model.parameters():
                p.data.uniform_(-model_opt.param_init, model_opt.param_init)
            for generator in generators.values():
                for p in generator.parameters():
                    p.data.uniform_(-model_opt.param_init, model_opt.param_init)
        if model_opt.param_init_glorot:
            for p in model.parameters():
                if p.dim() > 1:
                    xavier_uniform_(p)
            for generator in generators.values():
                for p in generator.parameters():
                    if p.dim() > 1:
                        xavier_uniform_(p)

    n_params, enc, dec = _tally_parameters(model)
    logger.info('encoder: %d' % enc)
    logger.info('decoder: %d' % dec)
    logger.info('* number of parameters: %d' % n_params)
    logger.info(model)

    _check_save_model_path(opt)

    # Build optimizer.
    optim = build_optim(model, opt, checkpoint)

    # Build model saver
    model_saver = build_model_saver(model_opt, opt, model, fields, optim)

    trainer = build_trainer(
        opt, model, fields, optim, data_type, generators, tgt_vocabs,
        model_saver=model_saver)

    trainer.train(train_iter_fcts, valid_iter_fcts, opt.train_steps,
                  opt.valid_steps)

    if opt.tensorboard:
        trainer.report_manager.tensorboard_writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='train.py',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    opts.add_md_help_argument(parser)
    opts.model_opts(parser)
    opts.train_opts(parser)

    opt = parser.parse_args()
    main(opt)
