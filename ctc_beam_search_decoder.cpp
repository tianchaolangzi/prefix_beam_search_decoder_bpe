#include "ctc_beam_search_decoder.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <map>
#include <utility>

#include "ThreadPool.h"
#include "fst/fstlib.h"

#include "decoder_utils.h"
#include "path_trie.h"

using FSTMATCH = fst::SortedMatcher<fst::StdVectorFst>;

std::vector<std::pair<double, std::string>> ctc_beam_search_decoder(
    const std::vector<std::vector<double>> &probs_seq,
    const std::vector<std::string> &vocabulary,
    size_t beam_size,
    double cutoff_prob,
    size_t cutoff_top_n,
    Scorer *ext_scorer) {
  // dimension check
  std::vector<std::tuple<std::string, uint32_t, uint32_t>> wordlist;
  size_t num_time_steps = probs_seq.size();
  for (size_t i = 0; i < num_time_steps; ++i) {
    VALID_CHECK_EQ(probs_seq[i].size(),
                   vocabulary.size() + 1,
                   "The shape of probs_seq does not match with "
                   "the shape of the vocabulary");
  }

  // assign blank id
  size_t blank_id = vocabulary.size();

  // init prefixes' root
  PathTrie root;
  root.score = root.log_prob_b_prev = 0.0;
  std::vector<PathTrie *> prefixes;
  prefixes.push_back(&root);

  if (ext_scorer != nullptr && !ext_scorer->is_character_based()) {
    auto fst_dict = static_cast<fst::StdVectorFst *>(ext_scorer->dictionary);
    fst::StdVectorFst *dict_ptr = fst_dict->Copy(true);
    root.set_dictionary(dict_ptr);
    auto matcher = std::make_shared<FSTMATCH>(*dict_ptr, fst::MATCH_INPUT);
    root.set_matcher(matcher);
  }

  // prefix search over time
  for (size_t time_step = 0; time_step < num_time_steps; ++time_step) {
    auto &prob = probs_seq[time_step];

    float min_cutoff = -NUM_FLT_INF;
    bool full_beam = false;
    if (ext_scorer != nullptr) {
      size_t num_prefixes = std::min(prefixes.size(), beam_size);
      std::sort(
          prefixes.begin(), prefixes.begin() + num_prefixes, prefix_compare);
      min_cutoff = prefixes[num_prefixes - 1]->score +
                   std::log(prob[blank_id]) - std::max(0.0, ext_scorer->beta);
      full_beam = (num_prefixes == beam_size);
    }

    std::vector<std::pair<size_t, float>> log_prob_idx =
        get_pruned_log_probs(prob, cutoff_prob, cutoff_top_n);
    // loop over chars
    for (size_t index = 0; index < log_prob_idx.size(); index++) {
      auto c = log_prob_idx[index].first;
      auto log_prob_c = log_prob_idx[index].second;
      // ????????????token????????????word??????????????????????????????????????????word
      bool word_end = false;
      if (c < vocabulary.size()) {
        std::string token = vocabulary[c];
        if (token.substr(0, 1) != "#"){
          word_end = true;
        }
      }
      for (size_t i = 0; i < prefixes.size() && i < beam_size; ++i) {
        auto prefix = prefixes[i];
        if (full_beam && log_prob_c + prefix->score < min_cutoff) {
          break;
        }
        
        // blank
        if (c == blank_id) {
          prefix->log_prob_b_cur =
              log_sum_exp(prefix->log_prob_b_cur, log_prob_c + prefix->score);
          continue;
        }
        // repeated character
        if (c == prefix->character) {
          prefix->log_prob_nb_cur = log_sum_exp(
              prefix->log_prob_nb_cur, log_prob_c + prefix->log_prob_nb_prev);
        }
        // get new prefix
        // ?????????????????????????????????token???????????????????????????????????????
        auto prefix_new = prefix->get_path_trie(c, word_end);

        // ?????????????????????????????????????????????????????????prefix???????????????
        if (prefix_new != nullptr) {
          float log_p = -NUM_FLT_INF;

          // ????????????token?????????????????????????????????token?????????????????????????????????ctc?????????blank???????????????
          // ??????????????????????????? eg. ab_b -> abb
          if (c == prefix->character &&
              prefix->log_prob_b_prev > -NUM_FLT_INF) {
            log_p = log_prob_c + prefix->log_prob_b_prev;
          } else if (c != prefix->character) {
            log_p = log_prob_c + prefix->score;
          }

          // language model scoring
          
          // ??????????????????????????????word????????????n-gram???score
          if (ext_scorer != nullptr && prefix->character != -1 &&
              (word_end || ext_scorer->is_character_based())) {
            PathTrie *prefix_to_score = nullptr;
            // skip scoring the space
            if (ext_scorer->is_character_based()) {
              prefix_to_score = prefix_new;
            } else {
              // prefix_to_score = prefix;
              // ???subword?????????????????????token??????????????????????????????
              // ????????????????????????prefix??????
              prefix_to_score = prefix;
            }

            float score = 0.0;
            std::vector<std::string> ngram;
            ngram = ext_scorer->make_ngram(prefix_to_score);
            
            score = ext_scorer->get_log_cond_prob(ngram) * ext_scorer->alpha;
            log_p += score;
            log_p += ext_scorer->beta;
          }
          prefix_new->log_prob_nb_cur =
              log_sum_exp(prefix_new->log_prob_nb_cur, log_p);
        }
      }  // end of loop over prefix
    }    // end of loop over vocabulary


    prefixes.clear();
    // update log probs
    root.iterate_to_vec(prefixes);

    // only preserve top beam_size prefixes
    if (prefixes.size() >= beam_size) {
      std::nth_element(prefixes.begin(),
                       prefixes.begin() + beam_size,
                       prefixes.end(),
                       prefix_compare);
      for (size_t i = beam_size; i < prefixes.size(); ++i) {
        prefixes[i]->remove();
      }
    }
  }  // end of loop over time

  // score the last word of each prefix that doesn't end with space
  if (ext_scorer != nullptr && !ext_scorer->is_character_based()) {
    for (size_t i = 0; i < beam_size && i < prefixes.size(); ++i) {
      auto prefix = prefixes[i];
      if (!prefix->is_empty()) {
        float score = 0.0;
        std::vector<std::string> ngram = ext_scorer->make_ngram(prefix);
        score = ext_scorer->get_log_cond_prob(ngram) * ext_scorer->alpha;
        score += ext_scorer->beta;
        prefix->score += score;
      }
    }
  }

  size_t num_prefixes = std::min(prefixes.size(), beam_size);
  std::sort(prefixes.begin(), prefixes.begin() + num_prefixes, prefix_compare);

  // compute aproximate ctc score as the return score, without affecting the
  // return order of decoding result. To delete when decoder gets stable.
  for (size_t i = 0; i < beam_size && i < prefixes.size(); ++i) {
    double approx_ctc = prefixes[i]->score;
    if (ext_scorer != nullptr) {
      std::vector<int> output;
      prefixes[i]->get_path_vec2(output, vocabulary);
      auto prefix_length = output.size();
      auto words = ext_scorer->split_labels(output);
      // remove word insert
      approx_ctc = approx_ctc - prefix_length * ext_scorer->beta;
      // remove language model weight:
      approx_ctc -= (ext_scorer->get_sent_log_prob(words)) * ext_scorer->alpha;
    }
    prefixes[i]->approx_ctc = approx_ctc;
  }

  return get_beam_search_result(prefixes, vocabulary, beam_size, wordlist);
}



/*
class BeamDecoder {
public:
  BeamDecoder(const std::vector<std::string> &vocabulary,
         size_t beam_size,
         double cutoff_prob = 1.0,
         size_t cutoff_top_n = 40,
         Scorer *ext_scorer = nullptr);
  ~BeamDecoder();

  // decode a frame
  std::vector<std::pair<double, std::string>> decode(const std::vector<std::vector<double>> &probs_seq);

  // reset state
  void reset();

private:
  Scorer *ext_scorer;
  size_t beam_size;
  double cutoff_prob;
  size_t cutoff_top_n;

  // state
  std::vector<std::string> vocabulary;
  size_t blank_id;
  int space_id;

  PathTrie *root;
  std::vector<PathTrie *> prefixes;
}
*/



BeamDecoder::BeamDecoder(const std::vector<std::string> &vocabulary,
         size_t beam_size,
         double cutoff_prob,
         size_t cutoff_top_n,
         Scorer *ext_scorer)
{
  this->beam_size = beam_size;
  this->cutoff_prob = cutoff_prob;
  this->cutoff_top_n = cutoff_top_n;
  this->ext_scorer = ext_scorer;

  this->vocabulary = vocabulary;
  this->root = nullptr;

  // assign blank id
  blank_id = vocabulary.size()-1;

  // assign space id
  auto it = std::find(vocabulary.begin(), vocabulary.end(), " ");
  space_id = it - vocabulary.begin();
  // if no space in vocabulary
  if ((size_t)space_id >= vocabulary.size()) {
    space_id = -2;
  }

  reset();
}


BeamDecoder::~BeamDecoder()
{
  if (root != nullptr) {
    delete root;
  }
}


void BeamDecoder::reset(bool keep_offset /*default = false*/, bool keep_words /*default = false*/)
{
  // init prefixes' root
  if (root != nullptr) {
    delete root;
  }
  root = new PathTrie();
  root->score = root->log_prob_b_prev = 0.0;
  
  prefixes.clear();
  prefixes.push_back(root);

  if (ext_scorer != nullptr && !ext_scorer->is_character_based()) {
    auto fst_dict = static_cast<fst::StdVectorFst *>(ext_scorer->dictionary);
    fst::StdVectorFst *dict_ptr = fst_dict->Copy(true);
    root->set_dictionary(dict_ptr);
    auto matcher = std::make_shared<FSTMATCH>(*dict_ptr, fst::MATCH_INPUT);
    root->set_matcher(matcher);
  }

  if (keep_offset) {
    prev_time_offset += last_decoded_timestep + time_offset;
  } else {
    prev_time_offset = 0;
  }

  if (keep_words) {
    prev_wordlist.insert(
        std::end(prev_wordlist), std::begin(wordlist),
        std::end(wordlist));
  } else {
    prev_wordlist.clear();
  }

  wordlist.clear();
  time_offset = 0;
  last_decoded_timestep = 0;
}


std::vector<std::pair<double, std::string>> BeamDecoder::decode(const std::vector<std::vector<double>> &probs_seq)
{
  // dimension check
  size_t num_time_steps = probs_seq.size();
  for (size_t i = 0; i < num_time_steps; ++i) {
    VALID_CHECK_EQ(probs_seq[i].size(),
                   vocabulary.size(),
                   "The shape of probs_seq does not match with "
                   "the shape of the vocabulary");
  }

  // prefix search over time
  for (size_t time_step = 0; time_step < num_time_steps; ++time_step) {
    auto &prob = probs_seq[time_step];

    float min_cutoff = -NUM_FLT_INF;
    bool full_beam = false;

    // TODO: move sorting to the end of loop
    if (ext_scorer != nullptr) {
      size_t num_prefixes = std::min(prefixes.size(), beam_size);
      std::sort(
          prefixes.begin(), prefixes.begin() + num_prefixes, prefix_compare);
      min_cutoff = prefixes[num_prefixes - 1]->score +
                   std::log(prob[blank_id]) - std::max(0.0, ext_scorer->beta);
      full_beam = (num_prefixes == beam_size);
    }

    std::vector<std::pair<size_t, float>> log_prob_idx =
        get_pruned_log_probs(prob, cutoff_prob, cutoff_top_n);
    // loop over chars
    for (size_t index = 0; index < log_prob_idx.size(); index++) {
      auto c = log_prob_idx[index].first;
      auto log_prob_c = log_prob_idx[index].second;

      for (size_t i = 0; i < prefixes.size() && i < beam_size; ++i) {
        auto prefix = prefixes[i];
        if (full_beam && log_prob_c + prefix->score < min_cutoff) {
          break;
        }
        // blank
        if (c == blank_id) {
          prefix->log_prob_b_cur =
              log_sum_exp(prefix->log_prob_b_cur, log_prob_c + prefix->score);
          continue;
        }
        // repeated character
        if (c == prefix->character) {
          prefix->log_prob_nb_cur = log_sum_exp(
              prefix->log_prob_nb_cur, log_prob_c + prefix->log_prob_nb_prev);
        }
        // get new prefix
        auto prefix_new = prefix->get_path_trie(c);

        if (prefix_new != nullptr) {
          float log_p = -NUM_FLT_INF;
          prefix_new->offset = prev_time_offset + time_offset + time_step;

          if (c == prefix->character &&
              prefix->log_prob_b_prev > -NUM_FLT_INF) {
            log_p = log_prob_c + prefix->log_prob_b_prev;
          } else if (c != prefix->character) {
            log_p = log_prob_c + prefix->score;
          }

          // language model scoring
          if (ext_scorer != nullptr &&
              (c == space_id || ext_scorer->is_character_based())) {
            PathTrie *prefix_to_score = nullptr;
            // skip scoring the space
            if (ext_scorer->is_character_based()) {
              prefix_to_score = prefix_new;
            } else {
              prefix_to_score = prefix;
            }

            float score = 0.0;
            std::vector<std::string> ngram;
            ngram = ext_scorer->make_ngram(prefix_to_score);
            score = ext_scorer->get_log_cond_prob(ngram) * ext_scorer->alpha;
            log_p += score;
            log_p += ext_scorer->beta;
          }
          prefix_new->log_prob_nb_cur =
              log_sum_exp(prefix_new->log_prob_nb_cur, log_p);
        }
      }  // end of loop over prefix
    }    // end of loop over vocabulary

    prefixes.clear();
    // update log probs
    root->iterate_to_vec(prefixes);

    // only preserve top beam_size prefixes
    if (prefixes.size() >= beam_size) {
      std::nth_element(prefixes.begin(),
                       prefixes.begin() + beam_size,
                       prefixes.end(),
                       prefix_compare);
      for (size_t i = beam_size; i < prefixes.size(); ++i) {
        prefixes[i]->remove();
      }
    }
  }  // end of loop over time

  // TODO: remove sorting here
  size_t num_prefixes = std::min(prefixes.size(), beam_size);
  std::sort(prefixes.begin(), prefixes.begin() + num_prefixes, prefix_compare);
  last_decoded_timestep = num_time_steps;

  return get_beam_search_result(prefixes, vocabulary, beam_size, wordlist);
}

void BeamDecoder::get_word_timestamps(
    std::vector<std::tuple<std::string, uint32_t, uint32_t>>& words)
{
  words.clear();
  words.insert(std::end(words), std::begin(prev_wordlist),
      std::end(prev_wordlist));
  words.insert(
      std::end(words), std::begin(wordlist), std::end(wordlist));
}


std::vector<std::vector<std::pair<double, std::string>>>
ctc_beam_search_decoder_batch(
    const std::vector<std::vector<std::vector<double>>> &probs_split,
    const std::vector<std::string> &vocabulary,
    size_t beam_size,
    size_t num_processes,
    double cutoff_prob,
    size_t cutoff_top_n,
    Scorer *ext_scorer) {
  VALID_CHECK_GT(num_processes, 0, "num_processes must be nonnegative!");
  // thread pool
  ThreadPool pool(num_processes);
  // number of samples
  size_t batch_size = probs_split.size();

  // enqueue the tasks of decoding
  std::vector<std::future<std::vector<std::pair<double, std::string>>>> res;
  for (size_t i = 0; i < batch_size; ++i) {
    res.emplace_back(pool.enqueue(ctc_beam_search_decoder,
                                  probs_split[i],
                                  vocabulary,
                                  beam_size,
                                  cutoff_prob,
                                  cutoff_top_n,
                                  ext_scorer));
  }

  // get decoding results
  std::vector<std::vector<std::pair<double, std::string>>> batch_results;
  for (size_t i = 0; i < batch_size; ++i) {
    batch_results.emplace_back(res[i].get());
  }
  return batch_results;
}


