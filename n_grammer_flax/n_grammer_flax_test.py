"""Tests for n_grammer_flax"""

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
from jax.config import config
import jax
from flax.core import freeze, unfreeze
import torch

from n_grammer_flax import get_bigram_ids, PQNgrammer
from n_grammer_pytorch import VQNgrammer as ngrammer_torch

def to_np(x):
  """Converts TF/JAX tensors to numpy."""
  return np.asarray(x, dtype=np.float32)

TEST_CASE_GET_BIGRAM_IDS =[
    [
        {'ids': np.array([[[0],[3],[3],[2],[1], [2],[3], [2]]], dtype="float32"),
        'vocab_size': 4},
        np.array([[[0],[3],[15],[14],[ 9],[ 6],[11],[14]]])
        ],
    [
        {'ids': np.array([[[0],[3],[3],[2],[1], [2],[3], [2]]],dtype="float32"),
        'vocab_size': 4,
        'segment_pos': np.array([[0,0,0,1,1,1,2,2]])},
        np.array([[[0],[3],[3],[14],[9],[6],[11],[14]]])
        ]
 ] 

class NgrammerTest(parameterized.TestCase):
    def setUp(self):
        super().setUp()
        np.random.seed(123456)
        config.update("jax_enable_x64", True)

    @parameterized.parameters(
        (10000),
        (1000),
        (320000),
        (500),)
    def test_get_bigram_ids_max(self, vocab_size):
        ids = np.random.randint(vocab_size, size=(2,2,16), dtype=np.int64)
        ngram_ids = get_bigram_ids(ids, vocab_size)
        np_ngram_ids = to_np(ngram_ids)
        self.assertLess(np.max(np_ngram_ids), vocab_size**2)
    
    @parameterized.parameters(
      (10000),
      (1000),
      (320000),
      (500),)
    def test_get_bigram_ids_with_packing(self, vocab_size):
        ids = np.random.randint(vocab_size, size=(2,8,1), dtype=np.int64)
        segment_pos = np.array([[0, 1, 2, 3, 0, 1, 2, 3], [0, 1, 2, 0, 1, 2, 3, 4]])
        ngram_ids = get_bigram_ids(ids, vocab_size, segment_pos)
        np_ngram_ids = to_np(ngram_ids)
        self.assertLess(np.max(np_ngram_ids), vocab_size**2)
        self.assertEqual(np_ngram_ids[0, 0], ids[0, 0])
        self.assertEqual(np_ngram_ids[1, 0], ids[1, 0])
        self.assertEqual(np_ngram_ids[0, 4], ids[0, 4])
        self.assertEqual(np_ngram_ids[1, 3], ids[1, 3])

    @parameterized.parameters(TEST_CASE_GET_BIGRAM_IDS)
    def test_get_bigram_ids_result(self, input_param, expected_val):
        result = get_bigram_ids(**input_param)
        np.testing.assert_allclose(to_np(result), expected_val, atol=1e-4)
    
    @parameterized.parameters(
        (1024,768 * 256, 16, 16, 32, True),
        (16, 16, 8, 2, 32, True),
        (24, 24, 4, 4, 16, True),
        (32, 32, 16, 1, 64, True),
        (25, 25, 4, 2, 8, True))
    def test_equivalence_with_n_grammer_pytorch(self, num_clusters, ngram_vocab_size, ngram_emb_dim,
                                                num_heads, dim_per_head, concat_ngrams):
        """test result against pytorch implementation https://github.com/lucidrains/n-grammer-pytorch."""
        x = torch.rand(2, 1024, dim_per_head * num_heads)
        init_rngs = {'params': jax.random.PRNGKey(1), 'batch_stats': jax.random.PRNGKey(2)}
        model_t = ngrammer_torch(
            num_clusters = num_clusters,
            dim_per_head = dim_per_head,           
            num_heads = num_heads,
            ngram_vocab_size = ngram_vocab_size,
            ngram_emb_dim =ngram_emb_dim,
            concat_ngrams = concat_ngrams,
            decay = 0.999)
        model_f = PQNgrammer(
            num_clusters = num_clusters,
            num_heads = num_heads,
            dim_per_head = dim_per_head,
            ngram_vocab_size = ngram_vocab_size,
            ngram_emb_dim= ngram_emb_dim,
            concat_ngrams = concat_ngrams,
            decay = 0.999)
        init_variables = model_f.init(init_rngs, to_np(x))
        init_variables = init_variables.unfreeze()
        init_variables['batch_stats']['ProductQuantization_0']['means'] = to_np(model_t.vq.means.detach().clone())
        init_variables['params']['Ngrammer_0']['Embed_0']['embedding'] = to_np(model_t.ngram.ngram_embeds.weight.detach().clone())
        init_variables = freeze(init_variables)
        out_f,variables = model_f.apply(init_variables, to_np(x), mutable=['batch_stats'])
        out_t = model_t(x).detach()
        np.testing.assert_allclose(to_np(out_t), to_np(out_f), rtol=1e-3, atol=1e-5)
        np.testing.assert_allclose(
            to_np(model_t.vq.means.detach()),
            to_np(jax.tree_leaves(variables)[0]),rtol=1e-2, atol=1e-5)

if __name__ == '__main__':
    absltest.main()