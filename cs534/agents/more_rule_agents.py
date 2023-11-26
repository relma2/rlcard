from typing import TYPE_CHECKING
from collections import OrderedDict

if TYPE_CHECKING:
    from rlcard.core import Card

from typing import List

import numpy as np

import rlcard

from rlcard.models.model import Model

from rlcard.games.gin_rummy.utils.action_event import *

import rlcard.games.gin_rummy.utils.melding as melding
import rlcard.games.gin_rummy.utils.utils as utils


class GinRummyAlwaysReduceRuleAgent(object):
    '''
        Agent always discards highest deadwood value card and always picks up a card if it is better.
    '''

    def __init__(self):
        self.use_raw = False  # FIXME: should this be True ?

    @staticmethod
    def step(state):
        ''' Predict the action given the current state.
            Novice strategy:
                Case where can gin:
                    Choose one of the gin actions.
                Case where can knock:
                    Choose one of the knock actions.
                Case where can discard:
                    Gin if can. Knock if can.
                    Otherwise, put aside cards in some best meld cluster.
                    Choose one of the remaining cards with highest deadwood value.
                    Discard that card.
                Case otherwise:
                    Choose a random action.

        Args:
            state (numpy.array): an numpy array that represents the current state

        Returns:
            action (int): the action predicted
        '''
        legal_actions = state['legal_actions']
        actions = legal_actions.copy()
        if draw_card_action_id in actions and pick_up_discard_action_id in actions:
            hand = utils.decode_cards(state['obs'][0])
            current_deadwood = utils.get_deadwood_count(hand, melding.get_best_meld_clusters(hand))

            hand.append(utils.decode_cards(state['obs'][1])[0])
            for discard_card in hand:
                next_hand = [card for card in hand if card != discard_card]
                meld_clusters = melding.get_meld_clusters(hand=next_hand)
                for meld_cluster in meld_clusters:
                    deadwood_count = utils.get_deadwood_count(hand=next_hand, meld_cluster=meld_cluster)
                    if deadwood_count < current_deadwood:
                        return pick_up_discard_action_id
            
            return draw_card_action_id


        legal_action_events = [ActionEvent.decode_action(x) for x in legal_actions]
        gin_action_events = [x for x in legal_action_events if isinstance(x, GinAction)]
        knock_action_events = [x for x in legal_action_events if isinstance(x, KnockAction)]
        discard_action_events = [x for x in legal_action_events if isinstance(x, DiscardAction)]
        if gin_action_events:
            actions = [x.action_id for x in gin_action_events]
        elif knock_action_events:
            actions = [x.action_id for x in knock_action_events]
        elif discard_action_events:
            best_discards = GinRummyAlwaysReduceRuleAgent._get_best_discards(discard_action_events=discard_action_events,
                                                                       state=state)
            if best_discards:
                actions = [DiscardAction(card=card).action_id for card in best_discards]
        if type(actions) == OrderedDict:
            actions = list(actions.keys())
        return np.random.choice(actions)

    def eval_step(self, state):
        ''' Predict the action given the current state for evaluation.
            Since the agents is not trained, this function is equivalent to step function.

        Args:
            state (numpy.array): an numpy array that represents the current state

        Returns:
            action (int): the action predicted by the agent
            probabilities (list): The list of action probabilities
        '''
        probabilities = []
        return self.step(state), probabilities

    @staticmethod
    def _get_best_discards(discard_action_events, state) -> List[Card]:
        best_discards = []  # type: List[Card]
        final_deadwood_count = 999
        env_hand = state['obs'][0]
        hand = utils.decode_cards(env_cards=env_hand)
        for discard_action_event in discard_action_events:
            discard_card = discard_action_event.card
            next_hand = [card for card in hand if card != discard_card]
            meld_clusters = melding.get_meld_clusters(hand=next_hand)
            deadwood_counts = []
            for meld_cluster in meld_clusters:
                deadwood_count = utils.get_deadwood_count(hand=next_hand, meld_cluster=meld_cluster)
                deadwood_counts.append(deadwood_count)
            best_deadwood_count = min(deadwood_counts,
                                      default=utils.get_deadwood_count(hand=next_hand, meld_cluster=[]))
            if best_deadwood_count < final_deadwood_count:
                final_deadwood_count = best_deadwood_count
                best_discards = [discard_card]
            elif best_deadwood_count == final_deadwood_count:
                best_discards.append(discard_card)
        return best_discards


class GinRummyMeldRuleAgent(object):
    '''
        Agent always looks for melds.
    '''

    def __init__(self):
        self.use_raw = False  # FIXME: should this be True ?

    @staticmethod
    def step(state):
        legal_actions = state['legal_actions']
        actions = legal_actions.copy()
        if draw_card_action_id in actions and pick_up_discard_action_id in actions:
            hand = utils.decode_cards(state['obs'][0])
            current_length = GinRummyMeldRuleAgent._num_cards_in_meld_cluster(GinRummyMeldRuleAgent._get_longest_meld_clusters(melding.get_meld_clusters(hand)))

            hand.append(utils.decode_cards(state['obs'][1])[0])
            for discard_card in hand:
                next_hand = [card for card in hand if card != discard_card]
                meld_clusters = melding.get_meld_clusters(hand=next_hand)
                potential_length = GinRummyMeldRuleAgent._num_cards_in_meld_cluster(GinRummyMeldRuleAgent._get_longest_meld_clusters(meld_clusters))
                if potential_length > current_length:
                    return pick_up_discard_action_id
            
            return draw_card_action_id


        legal_action_events = [ActionEvent.decode_action(x) for x in legal_actions]
        gin_action_events = [x for x in legal_action_events if isinstance(x, GinAction)]
        knock_action_events = [x for x in legal_action_events if isinstance(x, KnockAction)]
        discard_action_events = [x for x in legal_action_events if isinstance(x, DiscardAction)]
        if gin_action_events:
            actions = [x.action_id for x in gin_action_events]
        elif knock_action_events:
            actions = [x.action_id for x in knock_action_events]
        elif discard_action_events:
            best_discards = GinRummyMeldRuleAgent._get_best_discards(discard_action_events=discard_action_events,
                                                                       state=state)
            if best_discards:
                actions = [DiscardAction(card=card).action_id for card in best_discards]
        if type(actions) == OrderedDict:
            actions = list(actions.keys())
        return np.random.choice(actions)

    def eval_step(self, state):
        ''' Predict the action given the current state for evaluation.
            Since the agents is not trained, this function is equivalent to step function.

        Args:
            state (numpy.array): an numpy array that represents the current state

        Returns:
            action (int): the action predicted by the agent
            probabilities (list): The list of action probabilities
        '''
        probabilities = []
        return self.step(state), probabilities

    @staticmethod
    def _get_best_discards(discard_action_events, state) -> List[Card]:
        best_discards = []  # type: List[Card]
        best_length = 0
        env_hand = state['obs'][0]
        hand = utils.decode_cards(env_cards=env_hand)
        for discard_action_event in discard_action_events:
            discard_card = discard_action_event.card
            next_hand = [card for card in hand if card != discard_card]
            meld_clusters = melding.get_meld_clusters(hand=next_hand)
            longest = GinRummyMeldRuleAgent._get_longest_meld_clusters(meld_clusters)
            length = GinRummyMeldRuleAgent._num_cards_in_meld_cluster(longest)
            if length > best_length:
                best_length = length
                best_discards = [discard_card]
            elif length == best_length:
                best_discards.append(discard_card)
        return best_discards
    
    @staticmethod
    def _get_longest_meld_clusters(meld_clusters):
        longest = 0
        longest_meld_clusters = []
        for meld_cluster in meld_clusters:
            length = GinRummyMeldRuleAgent._num_cards_in_meld_cluster(meld_cluster)
            if length == longest:
                longest_meld_clusters.append(meld_cluster)
            elif length > longest:
                longest = length
                longest_meld_clusters = [meld_cluster]
        return longest_meld_clusters
    
    @staticmethod
    def _num_cards_in_meld_cluster(meld_cluster):
        num = 0
        for meld in meld_cluster:
            num += len(meld)
        return num