from rlcard.games.gin_rummy.utils.action_event import *
import rlcard.games.gin_rummy.utils.melding as melding
import rlcard.games.gin_rummy.utils.utils as utils

def gin_rummy_rule_filter(legal_actions, state):
    filtered_actions = []
    discard_actions = []
    for action in legal_actions:
        if isinstance(ActionEvent.decode_action(action), DiscardAction):
            discard_actions.append(action)
        else:
            filtered_actions.append(action)

    filtered_actions += _get_unmelded_discards(discard_actions, state)

    return filtered_actions

def _get_unmelded_discards(discard_actions, state):
    discard_actions_set = set(discard_actions)
    unmelded_discard_actions = set()
    env_hand = state['obs'][0]
    hand = utils.decode_cards(env_cards=env_hand)

    meld_clusters = _get_best_meld_clusters(hand, melding.get_meld_clusters(hand=hand))
    if len(meld_clusters) == 0:
        return discard_actions
    for meld_cluster in meld_clusters:
        options = discard_actions_set.copy()
        for meld in meld_cluster:
            for card in meld:
                options.discard(discard_action_id + utils.get_card_id(card))

        for option in options:
            unmelded_discard_actions.add(option)

    return list(unmelded_discard_actions)

def _num_cards_in_meld_cluster(meld_cluster):
    num = 0
    for meld in meld_cluster:
        num += len(meld)
    return num

def _get_longest_meld_clusters(meld_clusters):
    longest = 0
    longest_meld_clusters = []
    for meld_cluster in meld_clusters:
        length = _num_cards_in_meld_cluster(meld_cluster)
        if length == longest:
            longest_meld_clusters.append(meld_cluster)
        elif length > longest:
            longest = length
            longest_meld_clusters = [meld_cluster]
    return longest_meld_clusters

def _get_best_meld_clusters(hand, meld_clusters):
    lowest_deadwood = 110
    best_meld_clusters = []
    for meld_cluster in meld_clusters:
        deadwood = _get_deadwood_count(hand, meld_cluster)
        if deadwood == lowest_deadwood:
            best_meld_clusters.append(meld_cluster)
        elif deadwood < lowest_deadwood:
            lowest_deadwood = deadwood
            best_meld_clusters = [meld_cluster]
    return best_meld_clusters

def _get_deadwood_count(hand, meld_cluster):
    meld_cards = [card for meld_pile in meld_cluster for card in meld_pile]
    deadwood = [card for card in hand if card not in meld_cards]
    deadwood_values = [utils.get_deadwood_value(card) for card in deadwood]
    return sum(deadwood_values)
