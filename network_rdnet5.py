import network_base

class RDNet5(network_base.BaseNetwork):
    def setup(self):
        (self.feed('data')
             .conv(7, 7, 64, 2, 2, name='conv1_7x7_s2')
             .max_pool(3, 3, 2, 2, name='pool1_3x3_s2')
             .lrn(2, 2e-05, 0.75, name='pool1_norm1')
             .conv(1, 1, 64, 1, 1, name='conv2_3x3_reduce')
             .conv(3, 3, 192, 1, 1, name='conv2_3x3')
             .lrn(2, 2e-05, 0.75, name='conv2_norm2')
             .max_pool(3, 3, 2, 2, name='pool2_3x3_s2')
             .conv(1, 1, 64, 1, 1, name='inception_3a_1x1'))

        (self.feed('pool2_3x3_s2')
             .conv(1, 1, 96, 1, 1, name='inception_3a_3x3_reduce')
             .conv(3, 3, 128, 1, 1, name='inception_3a_3x3'))

        (self.feed('pool2_3x3_s2')
             .conv(1, 1, 16, 1, 1, name='inception_3a_5x5_reduce')
             .conv(5, 5, 32, 1, 1, name='inception_3a_5x5'))

        (self.feed('pool2_3x3_s2')
             .max_pool(3, 3, 1, 1, name='inception_3a_pool')
             .conv(1, 1, 32, 1, 1, name='inception_3a_pool_proj'))

        (self.feed('inception_3a_1x1', 
                   'inception_3a_3x3', 
                   'inception_3a_5x5', 
                   'inception_3a_pool_proj')
             .concat(3, name='inception_3a_output')
             .conv(1, 1, 128, 1, 1, name='inception_3b_1x1'))

        (self.feed('inception_3a_output')
             .conv(1, 1, 128, 1, 1, name='inception_3b_3x3_reduce')
             .conv(3, 3, 192, 1, 1, name='inception_3b_3x3'))

        (self.feed('inception_3a_output')
             .conv(1, 1, 32, 1, 1, name='inception_3b_5x5_reduce')
             .conv(5, 5, 96, 1, 1, name='inception_3b_5x5'))

        (self.feed('inception_3a_output')
             .max_pool(3, 3, 1, 1, name='inception_3b_pool')
             .conv(1, 1, 64, 1, 1, name='inception_3b_pool_proj'))

        (self.feed('inception_3b_1x1', 
                   'inception_3b_3x3', 
                   'inception_3b_5x5', 
                   'inception_3b_pool_proj')
             .concat(3, name='inception_3b_output')
             .conv(1, 1, 192, 1, 1, name='inception_4a_1x1'))

        (self.feed('inception_3b_output')
             .conv(1, 1, 96, 1, 1, name='inception_4a_3x3_reduce')
             .conv(3, 3, 208, 1, 1, name='inception_4a_3x3'))

        (self.feed('inception_3b_output')
             .conv(1, 1, 16, 1, 1, name='inception_4a_5x5_reduce')
             .conv(5, 5, 48, 1, 1, name='inception_4a_5x5'))

        (self.feed('inception_3b_output')
             .max_pool(3, 3, 1, 1, name='inception_4a_pool')
             .conv(1, 1, 64, 1, 1, name='inception_4a_pool_proj'))

        (self.feed('inception_4a_1x1', 
                   'inception_4a_3x3', 
                   'inception_4a_5x5', 
                   'inception_4a_pool_proj')
             .concat(3, name='inception_4a_output')
             .conv(1, 1, 64, 1, 1, name='inception_4b_1x1_new'))

        (self.feed('inception_4a_output')
             .conv(1, 1, 112, 1, 1, name='inception_4b_3x3_reduce')
             .conv(3, 3, 64, 1, 1, name='inception_4b_3x3_new'))

        (self.feed('inception_4a_output')
             .conv(1, 1, 24, 1, 1, name='inception_4b_5x5_reduce')
             .conv(5, 5, 64, 1, 1, name='inception_4b_5x5'))

        (self.feed('inception_4a_output')
             .max_pool(3, 3, 1, 1, name='inception_4b_pool')
             .conv(1, 1, 64, 1, 1, name='inception_4b_pool_proj'))

        (self.feed('inception_4b_1x1_new', 
                   'inception_4b_3x3_new', 
                   'inception_4b_5x5', 
                   'inception_4b_pool_proj')
             .concat(3, name='inception_4b_output')
             .conv(3, 3, 256, 1, 1, name='conv4_3_CPM')
             .conv(3, 3, 64, 1, 1, name='conv4_4_CPM')
             .conv(3, 3, 128, 1, 1, name='conv5_1_CPM_L1')
             .conv(3, 3, 128, 1, 1, name='conv5_2_CPM_L1')
             .conv(3, 3, 128, 1, 1, name='conv5_3_CPM_L1')
             .conv(1, 1, 512, 1, 1, name='conv5_4_CPM_L1')
             .conv(1, 1, 40, 1, 1, relu=False, name='conv5_5_CPM_L1'))

        (self.feed('conv4_4_CPM')
             .conv(3, 3, 128, 1, 1, name='conv5_1_CPM_L2')
             .conv(3, 3, 128, 1, 1, name='conv5_2_CPM_L2')
             .conv(3, 3, 128, 1, 1, name='conv5_3_CPM_L2')
             .conv(1, 1, 512, 1, 1, name='conv5_4_CPM_L2')
             .conv(1, 1, 20, 1, 1, relu=False, name='conv5_5_CPM_L2'))

        (self.feed('conv5_5_CPM_L1', 
                   'conv5_5_CPM_L2', 
                   'conv4_4_CPM')
             .concat(3, name='concat_stage2')
             .conv(3, 3, 128, 1, 1, name='Mconv1_stage2_L1')
             .conv(3, 3, 128, 1, 1, name='Mconv2_stage2_L1')
             .conv(3, 3, 128, 1, 1, name='Mconv3_stage2_L1')
             .conv(3, 3, 128, 1, 1, name='Mconv4_stage2_L1')
             .conv(3, 3, 128, 1, 1, name='Mconv5_stage2_L1')
             .conv(1, 1, 128, 1, 1, name='Mconv6_stage2_L1')
             .conv(1, 1, 40, 1, 1, relu=False, name='Mconv7_stage2_L1'))

        (self.feed('concat_stage2')
             .conv(3, 3, 128, 1, 1, name='Mconv1_stage2_L2')
             .conv(3, 3, 128, 1, 1, name='Mconv2_stage2_L2')
             .conv(3, 3, 128, 1, 1, name='Mconv3_stage2_L2')
             .conv(3, 3, 128, 1, 1, name='Mconv4_stage2_L2')
             .conv(3, 3, 128, 1, 1, name='Mconv5_stage2_L2')
             .conv(1, 1, 128, 1, 1, name='Mconv6_stage2_L2')
             .conv(1, 1, 20, 1, 1, relu=False, name='Mconv7_stage2_L2'))

        (self.feed('Mconv7_stage2_L1', 
                   'Mconv7_stage2_L2', 
                   'conv4_4_CPM')
             .concat(3, name='concat_stage3')
             .conv(3, 3, 128, 1, 1, name='Mconv1_stage3_L1')
             .conv(3, 3, 128, 1, 1, name='Mconv2_stage3_L1')
             .conv(3, 3, 128, 1, 1, name='Mconv3_stage3_L1')
             .conv(3, 3, 128, 1, 1, name='Mconv4_stage3_L1')
             .conv(3, 3, 128, 1, 1, name='Mconv5_stage3_L1')
             .conv(1, 1, 128, 1, 1, name='Mconv6_stage3_L1')
             .conv(1, 1, 40, 1, 1, relu=False, name='Mconv7_stage3_L1'))

        (self.feed('concat_stage3')
             .conv(3, 3, 128, 1, 1, name='Mconv1_stage3_L2')
             .conv(3, 3, 128, 1, 1, name='Mconv2_stage3_L2')
             .conv(3, 3, 128, 1, 1, name='Mconv3_stage3_L2')
             .conv(3, 3, 128, 1, 1, name='Mconv4_stage3_L2')
             .conv(3, 3, 128, 1, 1, name='Mconv5_stage3_L2')
             .conv(1, 1, 128, 1, 1, name='Mconv6_stage3_L2')
             .conv(1, 1, 20, 1, 1, relu=False, name='Mconv7_stage3_L2'))


    def loss_l1_l2(self):
         l1s = []
         l2s = []
         for layer_name in self.layers.keys():
              if 'Mconv7' in layer_name and '_L1' in layer_name:
                   l1s.append(self.layers[layer_name])
              if 'Mconv7' in layer_name and '_L2' in layer_name:
                   l2s.append(self.layers[layer_name])

         return l1s, l2s

    def loss_last(self):
         return self.get_output('Mconv7_stage3_L1'), self.get_output('Mconv7_stage3_L2')

    def restorable_variables(self):
         return None