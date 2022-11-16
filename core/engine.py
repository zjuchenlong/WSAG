import pickle as pkl

class Engine(object):
    def __init__(self):
        self.hooks = {}

    def hook(self, name, state):

        if name in self.hooks:
            self.hooks[name](state)

    def train(self, network, iterator, maxepoch, optimizer, scheduler):
        state = {
            'network': network,
            'iterator': iterator,
            'maxepoch': maxepoch,
            'optimizer': optimizer,
            'scheduler': scheduler,
            'epoch': 0,
            't': 0,
            'train': True,
        }

        # self.hook('debug_state', state)
        # import pdb; pdb.set_trace()

        self.hook('on_start', state)
        while state['epoch'] < state['maxepoch']:
            self.hook('on_start_epoch', state)
            for sample in state['iterator']:
                state['sample'] = sample
                self.hook('on_sample', state)

                def closure():
                    if state['epoch'] < 3:
                        warming = True
                    else:
                        warming = False
                    loss, output, loss_value_dict = state['network'](state['sample'], warming)

                    # print(f"iter {state['t']}, loss: {state['loss_meter'].avg}")
                    # if config.tensorboard:
                    #     writter.add_scalar('Loss/train', loss.item(), state['t'])
                    state['output'] = output
                    state['loss'] = loss
                    state['loss_value_dict'] = loss_value_dict
                    self.hook('tensorboard_writter', state)
                    loss.backward()
                    self.hook('on_forward', state)
                    # to free memory in save_for_backward
                    state['output'] = None
                    state['loss'] = None
                    return loss

                state['optimizer'].zero_grad()
                state['optimizer'].step(closure)
                self.hook('on_update', state)
                state['t'] += 1
            state['epoch'] += 1
            self.hook('on_end_epoch', state)
        self.hook('on_end', state)
        return state

    def test(self, network, iterator, split):
        state = {
            'network': network,
            'iterator': iterator,
            'split': split,
            't': 0,
            'train': False,
        }

        self.hook('on_test_start', state)
        for sample in state['iterator']:
            state['sample'] = sample
            self.hook('on_test_sample', state)

            def closure():
                # loss, output = state['network'](state['sample'])
                loss, output, prediction, proposal = state['network'](state['sample'])
                state['output'] = output
                state['loss'] = loss
                state['prediction'] = prediction,
                state['proposal'] = proposal
                self.hook('on_test_forward', state)
                # to free memory in save_for_backward
                state['output'] = None
                state['loss'] = None

            closure()
            state['t'] += 1
        # pkl.dump(state['save_sample_prediction'], open('baselines/MILNCE-WSTAN/train.pkl', 'wb'))
        # import pdb; pdb.set_trace()
        self.hook('on_test_end', state)
        return state