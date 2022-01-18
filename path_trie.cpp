#include "path_trie.h"

#include <algorithm>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

#include "decoder_utils.h"

PathTrie::PathTrie() {
  log_prob_b_prev = -NUM_FLT_INF;
  log_prob_nb_prev = -NUM_FLT_INF;
  log_prob_b_cur = -NUM_FLT_INF;
  log_prob_nb_cur = -NUM_FLT_INF;
  score = -NUM_FLT_INF;

  ROOT_ = -1;
  character = ROOT_;
  exists_ = true;
  parent = nullptr;

  dictionary_ = nullptr;
  dictionary_state_ = 0;
  has_dictionary_ = false;
  offset = 0;

  matcher_ = nullptr;
}

PathTrie::~PathTrie() {
  for (auto child : children_) {
    delete child.second;
  }
}

PathTrie* PathTrie::get_path_trie(int new_char, bool reset) {
  auto child = children_.begin();
  for (child = children_.begin(); child != children_.end(); ++child) {
    if (child->first == new_char) {
      break;
    }
  }
  if (child != children_.end()) {
    if (!child->second->exists_) {
      child->second->exists_ = true;
      child->second->log_prob_b_prev = -NUM_FLT_INF;
      child->second->log_prob_nb_prev = -NUM_FLT_INF;
      child->second->log_prob_b_cur = -NUM_FLT_INF;
      child->second->log_prob_nb_cur = -NUM_FLT_INF;
    }
    return (child->second);
  } else {
    if (has_dictionary_) {
      if (reset){
        matcher_->SetState(dictionary_->Start());
      } else {
        matcher_->SetState(dictionary_state_);
      }
      bool found = matcher_->Find(new_char + 1);
      if (!found) {
        return nullptr;
      } else {
        PathTrie* new_path = new PathTrie;
        new_path->character = new_char;
        new_path->parent = this;
        new_path->dictionary_ = dictionary_;
        new_path->dictionary_state_ = matcher_->Value().nextstate;
        new_path->has_dictionary_ = true;
        new_path->matcher_ = matcher_;
        children_.push_back(std::make_pair(new_char, new_path));
        return new_path;
      }
    } else {
      PathTrie* new_path = new PathTrie;
      new_path->character = new_char;
      new_path->parent = this;
      children_.push_back(std::make_pair(new_char, new_path));
      return new_path;
    }
  }
}

PathTrie* PathTrie::get_path_vec2(std::vector<int>& output,
                                  const std::vector<std::string>& char_list,
                                  std::vector<uint32_t>* timestamps) {
  if (character == ROOT_) {
    std::reverse(output.begin(), output.end());
    if (timestamps) {
      std::reverse(timestamps->begin(), timestamps->end());
    }
    return this;
  } else {
    output.push_back(character);
    if (timestamps) {
      if (timestamps->size() == 0 || output[output.size()-1] == 0 || parent->character == ROOT_ || parent->character == 0) {
        timestamps->push_back(offset);
      }
    }
    return parent->get_path_vec2(output, char_list, timestamps);
  }
}


PathTrie* PathTrie::get_path_vec(std::vector<int>& output,
                                 std::vector<std::string>& char_list,
                                 size_t max_steps,
                                 std::vector<uint32_t>* timestamps) {
  // TODO 如果当前路径的token为start_token，
  // output.push_back(token)
  // return parent
  if (character == ROOT_ || output.size() == max_steps) {
    std::reverse(output.begin(), output.end());
    if (timestamps) {
      std::reverse(timestamps->begin(), timestamps->end());
    }
    return this;
  } else {
    std::string token = char_list[character];
    if (token.substr(0, 1) != "#") {
      output.push_back(character);
      std::reverse(output.begin(), output.end());
      if (timestamps) {
        if (timestamps->size() == 0 || output[output.size()-1] == 0 || parent->character == ROOT_ || parent->character == 0) {
          timestamps->push_back(offset);
        }
        std::reverse(timestamps->begin(), timestamps->end());
      }
      return parent;
    } else {
      output.push_back(character);
      if (timestamps) {
        if (timestamps->size() == 0 || output[output.size()-1] == 0 || parent->character == ROOT_ || parent->character == 0) {
          timestamps->push_back(offset);
        }
      }
      return parent->get_path_vec(output, char_list, max_steps, timestamps);
    }
  }
}

void PathTrie::iterate_to_vec(std::vector<PathTrie*>& output) {
  if (exists_) {
    log_prob_b_prev = log_prob_b_cur;
    log_prob_nb_prev = log_prob_nb_cur;

    log_prob_b_cur = -NUM_FLT_INF;
    log_prob_nb_cur = -NUM_FLT_INF;

    score = log_sum_exp(log_prob_b_prev, log_prob_nb_prev);
    output.push_back(this);
  }
  for (auto child : children_) {
    child.second->iterate_to_vec(output);
  }
}

void PathTrie::remove() {
  exists_ = false;

  if (children_.size() == 0) {
    auto child = parent->children_.begin();
    for (child = parent->children_.begin(); child != parent->children_.end();
         ++child) {
      if (child->first == character) {
        parent->children_.erase(child);
        break;
      }
    }

    if (parent->children_.size() == 0 && !parent->exists_) {
      parent->remove();
    }

    delete this;
  }
}

void PathTrie::set_dictionary(fst::StdVectorFst* dictionary) {
  dictionary_ = dictionary;
  dictionary_state_ = dictionary->Start();
  has_dictionary_ = true;
}

using FSTMATCH = fst::SortedMatcher<fst::StdVectorFst>;
void PathTrie::set_matcher(std::shared_ptr<FSTMATCH> matcher) {
  matcher_ = matcher;
}





