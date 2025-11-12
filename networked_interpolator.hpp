#ifndef NETWORKED_INTERPOLATOR_HPP
#define NETWORKED_INTERPOLATOR_HPP

#include <functional>
#include <deque>
#include <optional>

#include "sbpt_generated_includes.hpp"

/**
 * @brief A network-aware interpolator that smooths discrete, jittery network updates.
 *
 * This interpolator receives discrete state updates from a remote source (e.g., a game server)
 * at an *average* frequency R1, which can vary due to network latency and jitter. Locally, it
 * produces smooth interpolated states at a higher frequency R2 (e.g., render frame rate) while
 * maintaining a stable interpolation interval driven by a local clock.
 *
 * @note
 * - Incoming network updates are buffered immediately when received via register_new_state().
 *   Their arrival times may vary due to network conditions.
 * - The *active interpolation window* (the two states being interpolated between) is advanced
 *   **only when the local receive signal fires** (via PeriodicSignal). This ensures a stable,
 *   jitter-resistant interpolation cadence.
 * - If no new network state has arrived when the signal fires, the interpolator “holds” the last
 *   end state rather than extrapolating forward, preventing visible discontinuities.
 *
 * @note the term "interpolation window" is the active start/end state being used during interpolation
 *
 * @tparam StateToInterpolate The type of state being interpolated (e.g., GameUpdate, Transform, etc.).
 */
template <typename StateToInterpolate> class NetworkedInterpolator {
  public:
    using InterpolationFunc =
        std::function<StateToInterpolate(const StateToInterpolate &, const StateToInterpolate &, float)>;

    struct StartStateChangedSignal {
        StateToInterpolate state;
    };

    /**
     * @brief a signal that other non interpolated systems should bind to to synchronize with the interpolation
     *
     * Because interpolation imposes a delay on the rendering of the interpolated objects as we have to wait until we
     * have two states so that we can interpolate between them, then this means that if a system executed logic based on
     * states received from the server the moment that it was received, then there would be a synchronization issue
     * between this system and the interpolated state.
     *
     * Therefore in order for non-interpolated systems to say in sync with the interpolated state, we need to
     * synchronize them whenever the active start state changes, here's why that makes sense. Consider a sound system,
     * as mentioned above, if we just played sounds the moment they were received, then there would be an offset between
     * say a players interpolated position and the moment they make a sound when they jump, and the sound would occur
     * before you jump off the ground. Thus by playing the sound based on the active start state it allows the sound to
     * stay roughly in sync with the current interpolated state.
     *
     */
    SignalEmitter start_state_changed_synchronization;

    /**
     * @brief Construct a networked interpolator with a user-defined interpolation function and rate.
     *
     * @param interpolate_func Function defining how to interpolate between two states.
     * @param receive_frequency Frequency (Hz) representing the expected network tick rate.
     */
    NetworkedInterpolator(InterpolationFunc interpolate_func, float receive_frequency)
        : interpolate_func(interpolate_func), idealized_receive_signal(receive_frequency),
          first_interpolation_window_has_been_initialized(false) {}

    /**
     * @brief Register a new incoming network state.
     *
     * This adds the state to a buffer for future use. It does *not* immediately change
     * the currently active interpolation window. The window is updated only at the local
     * receive signal rate inside update_interpolated_state().
     */
    void register_new_state(const StateToInterpolate &new_state) {
        incoming_state_buffer.push_back(new_state);

        // prevent unbounded growth in pathological network conditions.
        constexpr std::size_t kMaxBufferedStates = 32;
        if (incoming_state_buffer.size() > kMaxBufferedStates)
            incoming_state_buffer.pop_front();
    }

    /**
     * @brief Update the interpolated state based on local interpolation progress.
     *
     * Should be called at the render or update frequency R2 (e.g., once per frame).
     *
     * Behavior:
     * - If not initialized and enough buffered states exist, initializes the active window.
     * - Interpolates between the active start and end states using the cycle progress.
     * - When the local receive signal fires:
     *     * If a buffered network state exists, consume it as the new active_end_state.
     *     * Otherwise, copy active_end_state to active_start_state to hold the last known state.
     */
    void update() {
        // initialize window if not yet and we have enough buffered states
        if (!first_interpolation_window_has_been_initialized) { // A
            bool can_create_first_interpolation_window = incoming_state_buffer.size() >= 2;
            if (can_create_first_interpolation_window) { // B
                active_start_state = incoming_state_buffer.front();
                incoming_state_buffer.pop_front();
                active_end_state = incoming_state_buffer.front();
                incoming_state_buffer.pop_front();
                first_interpolation_window_has_been_initialized = true;
            } else { // C
                // not enough data yet to interpolate, stop the update process, hopefully theres some data on the next
                // call
                return;
            }
        }

        // NOTE: the only way you cannot get here is when fiwhbi is false, therefore when fiwhbi is true, then you will
        // get here. The only way for fiwhbi to become true is by running the body of B, which makes both
        // active_start_state and active_end_state initialized and no longer nullopt, therefore the usage of .value() is
        // justified below.
        // NOTE:0: Also note that this implies that update_interpolation_window has this property as a
        // precondition because its only ever called in here

        // compute interpolation based on current progress
        active_t = idealized_receive_signal.get_cycle_progress(); // [0,1] through current local interval

        current_interpolated_state = interpolate_func(active_start_state.value(), active_end_state.value(), active_t);

        if (idealized_receive_signal.process_and_get_signal()) {
            update_interpolation_window();
        }
    }

    /**
     * @brief Updates the active state window based on the incoming state buffer.
     *
     * This function manages the interpolation window by updating `active_start_state`
     * and `active_end_state` according to the states available in `incoming_state_buffer`.
     *
     * Suppose we have some state updates sent from the server to client:
     *
     * s1, s2, s3, s4, ...
     *
     */
    void update_interpolation_window() {

        bool new_states_received_since_last_time_this_was_called = not incoming_state_buffer.empty();
        bool no_new_states_received_since_last_time_this_was_called =
            not new_states_received_since_last_time_this_was_called;

        if (new_states_received_since_last_time_this_was_called) { // A

            if (incoming_state_buffer.size() == 1) {
                // Normal update
                active_start_state = active_end_state;
                active_end_state = incoming_state_buffer.front();
                incoming_state_buffer.pop_front();
            } else {
                // catch-up: take the last two states
                // TODO: document the implications of the catch up method because it surely should cause some sort of
                // jitter.
                auto it = incoming_state_buffer.end();
                --it; // last element
                active_end_state = *it;
                --it; // second to last element
                active_start_state = *it;

                // Remove all processed states from buffer
                incoming_state_buffer.erase(incoming_state_buffer.begin(), incoming_state_buffer.end());
            }
        }

        // NOTE: once here is reached then the incoming state buffer is completely empty

        if (no_new_states_received_since_last_time_this_was_called) { // B
            // NOTE: this is a very important line, we only ever get here if we haven't registered a new state since
            // last time this was called, this occurs when the network variance causes one of the packets to take longer
            // than it regularly would. If we didn't add this line then the active start/end state so the interpolation
            // window is the same as the last interpolation window which would look strange because it would look like
            // you interpolate from point a to point b, and then do that again, this line fixes that because it makes a
            // "constant interpolation window" as the start and end state become the same, once this is true all
            // interpolation attempts always just yield active_end_state which is the most up to date state, which is
            // better than doing the same interpolation again.
            active_start_state = active_end_state;
        }

        // NOTE: this variable being true means that you haven't received an update from the server in two iterations,
        // this most likely means that there's some sort of internet problem going on, and here's how it effects the
        // interpolation window creation process. Lets say the server sent over some states s1, s2, s3, ... . On
        // your computer you receive s1, s2, s3 and then for some reason none of the other states get there for a
        // moment, then the sequence of interpolation windows created inside of this system is like this:
        // s1-s2 (A), s2-s3 (A), s3-s3 (B) , s3-s3 (B), s3-s3 (B), ...
        // note: the upper case characters represent which if statement above created the window
        // So in the above image we can see that the second time the (B) logic is used, the start state is the same
        // as it was in the previous or equivalently the interpolation window is duplicated. Whenever we are in this
        // state, then we should not emit the start state changed signal, or else for example the sound system would
        // repeatedly play the same sound over and over as its getting duplicated
        bool at_least_two_consecutive_empty_incoming_state_buffer_updates =
            incoming_state_buffer_was_empty_on_last_update and no_new_states_received_since_last_time_this_was_called;

        if (not at_least_two_consecutive_empty_incoming_state_buffer_updates) {
            start_state_changed_synchronization.emit(
                StartStateChangedSignal{active_start_state.value()}); // value usage justified by NOTE:0
        }

        // update for the next update
        incoming_state_buffer_was_empty_on_last_update = incoming_state_buffer.empty();
    }

    /// @brief Get the current interpolated state.
    const std::optional<StateToInterpolate> &get_state() const { return current_interpolated_state; }

    /// @brief Get the currently active "start" state being interpolated from.
    const std::optional<StateToInterpolate> &get_active_start_state() const { return active_start_state; }

    const double &get_active_t() const { return active_t; }

    /// @brief Get the currently active "end" state being interpolated toward.
    const std::optional<StateToInterpolate> &get_active_end_state() const { return active_end_state; }

  private:
    /// @brief Buffered incoming network states.
    std::deque<StateToInterpolate> incoming_state_buffer;

    bool incoming_state_buffer_was_empty_on_last_update = true; // initially there's nothing there

    /**
     * @brief The currently active start state (previous state in interpolation).
     *
     * Optional is used as this class is un-initialized before any states are registered, and allows us to safely access
     * this member, once the first call to update is when there is at least two elements then this is forever a real
     * value
     */
    std::optional<StateToInterpolate> active_start_state = std::nullopt;

    /// @brief The currently active t value (which is how far we are between the two states in time, normalized [0, 1])
    double active_t = 0;

    /**
     * @brief The currently active end state (next target state in interpolation).
     *
     * Optional is used as this class is un-initialized before any states are registered, and allows us to safely access
     * this member, once the first call to update is when there is at least two elements then this is forever a real
     * value
     */
    std::optional<StateToInterpolate> active_end_state = std::nullopt;

    /**
     * @brief The most recent interpolated result.
     *
     * Optional is used as this class is un-initialized before any states are registered, and allows us to safely access
     * this member, once the first call to update is when there is at least two elements then this is forever a real
     * value
     */
    std::optional<StateToInterpolate> current_interpolated_state = std::nullopt;

    /// @brief Whether the interpolation window is initialized. Only ever false at the start
    bool first_interpolation_window_has_been_initialized = false;

    InterpolationFunc interpolate_func;

    /**
     * @brief a signal which represents the client to server send rate, this allows us to keep a steady interpolation
     * window going, as the regular rate a which states are registered is a signal with the same average frequency but
     * more variance,
     *
     * @note that the idealized receive signal probably has a phase shift relative to the actual receive times, and
     * there is nothing we can do about that, and this probably doesn't have a huge impact, I just haven't thought about
     * how it might have an impact alot and wanted to mention it
     */
    PeriodicSignal idealized_receive_signal;
};

#endif // NETWORKED_INTERPOLATOR_HPP
