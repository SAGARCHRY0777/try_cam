"""
Event bus for manager level communications
"""
import asyncio
import threading
from typing import Callable, Dict, List, Any

class EventBus:
    """A lightweight thread-safe asynchronous event bus.

    This class allows different components to communicate by publishing and subscribing
    to named topics. Subscribers register callbacks that are automatically invoked
    when a message is published to a topic. Supports both synchronous and asynchronous
    subscriber callbacks.

    Thread safety is ensured using a reentrant lock to guard the subscriber registry.
    """
    def __init__(self):
        """Initialize a new EventBus instance.

        Attributes:
            _subscribers (Dict[str, List[Callable[[Any], Any]]]):
                A mapping of topic names to lists of subscriber callback functions.
            _lock (threading.RLock):
                A reentrant lock used to protect modifications to the subscriber registry.
        """
        self._subscribers: Dict[str, List[Callable[[Any], Any]]] = {}
        self._lock = threading.RLock()

    def subscribe(self, topic: str, callback: Callable[[Any], Any]) -> None:
        """Subscribe a callback function to a given topic.

        Args:
            topic (str): The name of the topic to subscribe to.
            callback (Callable[[Any], Any]): The function to call when a message is published
                to this topic. Can be synchronous or asynchronous.
        """
        with self._lock:
            self._subscribers.setdefault(topic, []).append(callback)

    def unsubscribe(self, topic: str, callback: Callable[[Any], Any]) -> None:
        """Unsubscribe a callback from a given topic.

        Args:
            topic (str): The name of the topic to unsubscribe from.
            callback (Callable[[Any], Any]): The callback function to remove.
        """
        with self._lock:
            if topic in self._subscribers:
                try:
                    self._subscribers[topic].remove(callback)
                except ValueError:
                    pass

    async def publish(self, topic: str, message: Any) -> None:
        """Publish a message to all subscribers of a given topic.

        Each subscriber callback will be invoked with the provided message.
        Asynchronous callbacks are scheduled with `asyncio.create_task`, while
        synchronous ones run in a background thread via `asyncio.to_thread`.

        Args:
            topic (str): The topic to publish the message to.
            message (Any): The data or event payload to send to subscribers.
        """
        with self._lock:
            subscribers = list(self._subscribers.get(topic, []))  # copy safely

        for callback in subscribers:
            if asyncio.iscoroutinefunction(callback):
                asyncio.create_task(callback(message))
            else:
                await asyncio.to_thread(callback, message)

# # Singleton instance for all managers
GLOBAL_EVENT_BUS = EventBus()
