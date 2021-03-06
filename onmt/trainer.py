"""
    This is the loadable seq2seq trainer library that is
    in charge of training details, loss compute, and statistics.
    See train.py for a use case of this library.

    Note: To make this a general library, we implement *only*
          mechanism things here(i.e. what to do), and leave the strategy
          things to users(i.e. how to do it). Also see train.py(one of the
          users of this library) for the strategy things we do.
"""

from __future__ import division

import random

from collections import OrderedDict

import onmt.inputters as inputters
import onmt.utils
import subprocess
import sys

from onmt.utils.loss import build_loss_from_generator_and_vocab

from onmt.utils.logging import logger
import torch
from torch.autograd import Variable

def build_trainer(opt, model, fields, optim, data_type,
                  generators,
                  tgt_vocabs,
                  model_saver=None):
    """
    Simplify `Trainer` creation based on user `opt`s*

    Args:
        opt (:obj:`Namespace`): user options (usually from argument parsing)
        model (:obj:`onmt.models.NMTModel`): the model to train
        fields (dict): dict of fields
        optim (:obj:`onmt.utils.Optimizer`): optimizer used during training
        data_type (str): string describing the type of data
            e.g. "text", "img", "audio"
        model_saver(:obj:`onmt.models.ModelSaverBase`): the utility object
            used to save the model
    """

    # Chris: one loss for every decoder
    train_losses = OrderedDict()
    valid_losses = OrderedDict()
    for tgt_lang, gen in generators.items():
        train_losses[tgt_lang] = \
            build_loss_from_generator_and_vocab(
                gen,
                tgt_vocabs[tgt_lang],
                opt,
                train=True
        )
        valid_losses[tgt_lang] = \
            build_loss_from_generator_and_vocab(
                gen,
                tgt_vocabs[tgt_lang],
                opt,
                train=False
            )

    trunc_size = opt.truncated_decoder  # Badly named...
    shard_size = opt.max_generator_batches
    norm_method = opt.normalization
    grad_accum_count = opt.accum_count
    n_gpu = len(opt.gpuid)
    gpu_rank = opt.gpu_rank
    gpu_verbose_level = opt.gpu_verbose_level

    report_manager = onmt.utils.build_report_manager(opt)
    trainer = onmt.Trainer(model, train_losses, valid_losses, optim, opt.attention_heads, trunc_size,
                           shard_size, data_type, norm_method,
                           grad_accum_count, n_gpu, gpu_rank,
                           gpu_verbose_level, report_manager,
                           opt.use_attention_bridge, model_saver=model_saver)
    return trainer


class Trainer(object):
    """
    Class that controls the training process.

    Args:
            model(:py:class:`onmt.models.model.NMTModel`): translation model
                to train
            train_loss(:obj:`onmt.utils.loss.LossComputeBase`):
               training loss computation
            valid_loss(:obj:`onmt.utils.loss.LossComputeBase`):
               training loss computation
            optim(:obj:`onmt.utils.optimizers.Optimizer`):
               the optimizer responsible for update
            trunc_size(int): length of truncated back propagation through time
            shard_size(int): compute loss in shards of this size for efficiency
            data_type(string): type of the source input: [text|img|audio]
            norm_method(string): normalization methods: [sents|tokens]
            grad_accum_count(int): accumulate gradients this many times.
            report_manager(:obj:`onmt.utils.ReportMgrBase`):
                the object that creates reports, or None
            model_saver(:obj:`onmt.models.ModelSaverBase`): the saver is
                used to save a checkpoint.
                Thus nothing will be saved if this parameter is None
    """

    def __init__(self, model, train_losses, valid_losses, optim, attention_heads,
                 trunc_size=0, shard_size=32, data_type='text',
                 norm_method="sents", grad_accum_count=1, n_gpu=1, gpu_rank=1,
                 gpu_verbose_level=0, report_manager=None, use_attention_bridge=True, model_saver=None):
        # Basic attributes.
        self.model = model
        self.train_losses = train_losses
        self.valid_losses = valid_losses
        self.optim = optim
        self.trunc_size = trunc_size
        self.shard_size = shard_size
        self.data_type = data_type
        self.norm_method = norm_method
        self.grad_accum_count = grad_accum_count
        self.n_gpu = n_gpu
        self.gpu_rank = gpu_rank
        self.gpu_verbose_level = gpu_verbose_level
        self.report_manager = report_manager
        self.model_saver = model_saver
        self.last_model = None
        self.use_attention_bridge = use_attention_bridge
        self.attention_heads = attention_heads

        assert grad_accum_count > 0
        if grad_accum_count > 1:
            assert(self.trunc_size == 0), \
                """To enable accumulated gradients,
                   you must disable target sequence truncating."""

        # Set model in training mode.
        self.model.train()

    # def train(self, train_iter_fct, valid_iter_fct, train_steps, valid_steps):
    def train(self, train_iter_fcts, valid_iter_fcts, train_steps, valid_steps):
        """
        The main training loops.
        by iterating over training data (i.e. `train_iter_fct`)
        and running validation (i.e. iterating over `valid_iter_fct`

        Args:
            train_iter_fct(function): a function that returns the train
                iterator. e.g. something like
                train_iter_fct = lambda: generator(*args, **kwargs)
            valid_iter_fct(function): same as train_iter_fct, for valid data
            train_steps(int):
            valid_steps(int):
            save_checkpoint_steps(int):

        Return:
            None
        """
        logger.info('Start training...')

        step = self.optim._step + 1
        true_batchs = []
        accum = 0
        normalization = 0

        total_stats = onmt.utils.Statistics()
        report_stats = onmt.utils.Statistics()
        self._start_report_manager(start_time=total_stats.start_time)

        # init every train iter
        train_iters = {k: (b for b in f())
                       for k, f in train_iter_fcts.items()}

        while step <= train_steps:

            reduce_counter = 0
            src_lang, tgt_lang = random.choice(list(train_iters.keys()))

            try:
                batch = next(train_iters[(src_lang, tgt_lang)])
            except:
                # re-init the iterator
                logger.info('recreating {}-{} dataset'.format(src_lang,
                                                              tgt_lang))
                train_iters[(src_lang, tgt_lang)] = \
                    (b for b in train_iter_fcts[(src_lang, tgt_lang)]())
                batch = next(train_iters[(src_lang, tgt_lang)])

            # assign the source and target langs to the batch
            setattr(batch, 'src_lang', src_lang)
            setattr(batch, 'tgt_lang', tgt_lang)

            # CHRIS: note this may not work yet for multi-gpu or accumulation
            #if self.n_gpu == 0 or (i % self.n_gpu == self.gpu_rank):
            if self.n_gpu == 0 or (step % self.n_gpu == self.gpu_rank):
                if self.gpu_verbose_level > 1:
                    logger.info("GpuRank %d: index: %d accum: %d"
                                % (self.gpu_rank, step, accum))

                true_batchs.append(batch)

                if self.norm_method == "tokens":
                    num_tokens = batch.tgt[1:].ne(
                        self.train_losses[tgt_lang].padding_idx).sum()
                    normalization += num_tokens.item()
                else:
                    normalization += batch.batch_size
                accum += 1
                if accum == self.grad_accum_count:
                    reduce_counter += 1
                    if self.gpu_verbose_level > 0:
                        logger.info("GpuRank %d: reduce_counter: %d \
                                    n_minibatch %d"
                                    % (self.gpu_rank, reduce_counter,
                                       len(true_batchs)))
                    if self.n_gpu > 1:
                        normalization = sum(onmt.utils.distributed
                                            .all_gather_list
                                            (normalization))

                    self._gradient_accumulation(
                        true_batchs, normalization, total_stats,
                        report_stats)

                    report_stats = self._maybe_report_training(
                        step, train_steps,
                        self.optim.learning_rate,
                        report_stats)

                    true_batchs = []
                    accum = 0
                    normalization = 0
                    if (step % valid_steps == 0):
                        for lang_pair in valid_iter_fcts.items():
                            valid_iter_fct = lang_pair[1]
                            src_tgt = lang_pair[0]
                            logger.info('Current language pair: {}'.format(src_tgt))
                            # loop valid over all lang pairs
                            if self.gpu_verbose_level > 0:
                                logger.info('GpuRank %d: validate step %d'
                                            % (self.gpu_rank, step))
                            valid_iter = valid_iter_fct()
                            valid_stats = self.validate(valid_iter, src_tgt)
                            if self.gpu_verbose_level > 0:
                                logger.info('GpuRank %d: gather valid stat \
                                            step %d' % (self.gpu_rank, step))
                            valid_stats = self._maybe_gather_stats(valid_stats)
                            if self.gpu_verbose_level > 0:
                                logger.info('GpuRank %d: report stat step %d'
                                            % (self.gpu_rank, step))
                            self._report_step(self.optim.learning_rate,
                                              step, valid_stats=valid_stats)

                        self._maybe_save(step)
                        """
                        # TODO: change in a better way
                        # method to have bleu score during validation: it translates on a file, and run a script in order to detokenize, untrucase, etc. and finally compute the bleu score 
                        
                        from onmt.translate.translator import build_translator
                        import argparse

                        for lang_pair in valid_iter_fcts.items():
                            src_tgt = lang_pair[0]
                            src_langDEV, tgt_langDEV = src_tgt
                            src_tmp = path/to/dev/text-file+str(src_langDEV)
                            out_tmp = path/to/dev/temp/output/+str(src_langDEV)+'-'+str(tgt_langDEV)+'.txt'

                            parser = argparse.ArgumentParser(prog='translate.py',
                                                             description='train.py')
                            onmt.opts.translate_opts(parser)
                            dummy_opt = parser.parse_known_args(['-model', self.last_model,
                                                                 '-src', src_tmp,
                                                                 '-output', out_tmp])[0]
                            dummy_opt.use_attention_bridge = self.use_attention_bridge

                            dummy_opt.src_lang, dummy_opt.tgt_lang = src_tgt
                            
                            translator = build_translator(dummy_opt, report_score=False)
                            translator.translate(src_path=dummy_opt.src,
                                                 tgt_path=dummy_opt.tgt,
                                                 src_dir=dummy_opt.src_dir,
                                                 batch_size=256,
                                                 attn_debug=False)
                            original = path/to/dev/text-file+str(src_langDEV)+str(dummy_opt.tgt_lang)+'.detok'
                            
                            res = subprocess.check_output("bash path/to/evaluation_script.sh %s %s %s" % (dummy_opt.tgt_lang, out_tmp, original), shell=True).decode("utf-8")
                            msg = res.strip()
                            #print(str(step)+" "+str(src_langDEV)+'-'+str(tgt_langDEV)+' '+str(msg), file=sys.stderr) # print on stderr for csc
                            print(str(step)+" "+str(src_langDEV)+'-'+str(tgt_langDEV)+' '+str(msg))
                        print()
                        """
                    step += 1
                    if step > train_steps:
                        break

            if self.gpu_verbose_level > 0:
                logger.info('GpuRank %d: we completed an epoch \
                            at step %d' % (self.gpu_rank, step))
            # train_iter = train_iter_fct()

        return total_stats

    def validate(self, valid_iter, src_tgt):
        """ Validate model.
            valid_iter: validate data iterator
        Returns:
            :obj:`nmt.Statistics`: validation loss statistics
        """
        # Set model in validating mode.
        self.model.eval()

        stats = onmt.utils.Statistics()

        for batch in valid_iter:
            setattr(batch, 'src_lang', src_tgt[0])
            setattr(batch, 'tgt_lang', src_tgt[1])
            src = inputters.make_features(batch, 'src', self.data_type)
            if self.data_type == 'text':
                _, src_lengths = batch.src
            else:
                src_lengths = None

            tgt = inputters.make_features(batch, 'tgt')

            # F-prop through the model.
            outputs, attns, _, alphasZ = self.model(src, tgt,
                            batch.src_lang,
                            batch.tgt_lang,
                            src_lengths)
            # Compute loss.
            batch_stats = \
                self.valid_losses[batch.tgt_lang].monolithic_compute_loss(
                    batch, outputs, attns)

            # Update statistics.
            stats.update(batch_stats)

        # Set model back to training mode.
        self.model.train()

        return stats

    def _gradient_accumulation(self, true_batchs, normalization, total_stats,
                               report_stats):
        if self.grad_accum_count > 1:
            self.model.zero_grad()

        I = Variable(torch.stack([torch.eye(self.attention_heads) for i in range(len(true_batchs[0]) ) ] )) #len(true_batchs[0] = true_batchs[0].__dict__['batch_size']
        I = I.cuda() if self.n_gpu >= 1 else I

        # Chris: the `batch` contains the information about what the source
        # Chris: and target languages are
        for batch in true_batchs:
            target_size = batch.tgt.size(0)
            # Truncated BPTT: reminder not compatible with accum > 1
            if self.trunc_size:
                trunc_size = self.trunc_size
            else:
                trunc_size = target_size

            dec_state = None
            src = inputters.make_features(batch, 'src', self.data_type)
            if self.data_type == 'text':
                _, src_lengths = batch.src
                report_stats.n_src_words += src_lengths.sum().item()
            else:
                src_lengths = None

            tgt_outer = inputters.make_features(batch, 'tgt')

            for j in range(0, target_size-1, trunc_size):
                # 1. Create truncated target.
                tgt = tgt_outer[j: j + trunc_size]

                # 2. F-prop all but generator.
                if self.grad_accum_count == 1:
                    self.model.zero_grad()

                outputs, attns, dec_state, alphasZ = \
                    self.model(src, tgt,
                               batch.src_lang,
                               batch.tgt_lang,
                               src_lengths,
                               dec_state)

                # 3. Compute loss in shards for memory efficiency.
                # Chris: note the loss is different for different decoders
                batch_stats = \
                    self.train_losses[batch.tgt_lang].sharded_compute_loss(
                        batch, outputs, attns, j,
                        trunc_size, self.shard_size, normalization, alphasZ, I)

                total_stats.update(batch_stats)
                report_stats.update(batch_stats)

                # 4. Update the parameters and statistics.
                if self.grad_accum_count == 1:
                    # Multi GPU gradient gather
                    if self.n_gpu > 1:
                        grads = [p.grad.data for p in self.model.parameters()
                                 if p.requires_grad
                                 and p.grad is not None]
                        onmt.utils.distributed.all_reduce_and_rescale_tensors(
                            grads, float(1))
                    self.optim.step()

                # If truncated, don't backprop fully.
                if dec_state is not None:
                    dec_state.detach()

        # in case of multi step gradient accumulation,
        # update only after accum batches
        if self.grad_accum_count > 1:
            if self.n_gpu > 1:
                grads = [p.grad.data for p in self.model.parameters()
                         if p.requires_grad
                         and p.grad is not None]
                onmt.utils.distributed.all_reduce_and_rescale_tensors(
                    grads, float(1))
            self.optim.step()

    def _start_report_manager(self, start_time=None):
        """
        Simple function to start report manager (if any)
        """
        if self.report_manager is not None:
            if start_time is None:
                self.report_manager.start()
            else:
                self.report_manager.start_time = start_time

    def _maybe_gather_stats(self, stat):
        """
        Gather statistics in multi-processes cases

        Args:
            stat(:obj:onmt.utils.Statistics): a Statistics object to gather
                or None (it returns None in this case)

        Returns:
            stat: the updated (or unchanged) stat object
        """
        if stat is not None and self.n_gpu > 1:
            return onmt.utils.Statistics.all_gather_stats(stat)
        return stat

    def _maybe_report_training(self, step, num_steps, learning_rate,
                               report_stats):
        """
        Simple function to report training stats (if report_manager is set)
        see `onmt.utils.ReportManagerBase.report_training` for doc
        """
        if self.report_manager is not None:
            return self.report_manager.report_training(
                step, num_steps, learning_rate, report_stats,
                multigpu=self.n_gpu > 1)

    def _report_step(self, learning_rate, step, train_stats=None,
                     valid_stats=None):
        """
        Simple function to report stats (if report_manager is set)
        see `onmt.utils.ReportManagerBase.report_step` for doc
        """
        if self.report_manager is not None:
            return self.report_manager.report_step(
                learning_rate, step, train_stats=train_stats,
                valid_stats=valid_stats)

    def _maybe_save(self, step):
        """
        Save the model if a model saver is set
        """
        if self.model_saver is not None:
            self.model_saver.maybe_save(step)
            self.last_model = self.model_saver.base_path + '_step_' + str(step) + '.pt'
