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

    meld_clusters = melding.get_meld_clusters(hand=hand)
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