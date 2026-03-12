This code takes voice input from the main microphone in use by the computer, and outputs the heard conversation in the terminal of the program. 

To do this, the program takes frames of all audio input, combines the frames, processes them, monitors the frames for an "intent" word/phrase and an "object" word/phrase. If the program hears both an intent and an object, it outputs that command in the terminal. However, if the program only hears the intent and not an object, or vice versa, it will output that it heard a keyword without a connecting keyword. 

Currently, the program does not have the robotic movement function. However, the speech output is still functional.

POTENTIAL PITFALLS:

  1. The program struggles with accuracy. It is assumed that a stronger version of Whisper (currently the program runs on 'small') or a better microphone would help this issue. While using the 'small' version of Whisper, it is advised to clearly annunciate any commands.

  2. If the program hears two library targets in one conversation frame, it will assume the first command. For instance, if the program hears, "Hey robot, grab that object and then drop it," it will only register the grab and object commands, and will disregard the drop command. Further refinement of the code will be needed to prevent this. Currently, it is recommended to ONLY command one intent and one object in a short span.

  3. The program hallucinates some connection wording, such as like, so, a, etc. This is considered a minor problem, as it still registers the command words.

BENEFITS:

  1. The program registers full commands and half commands, allowing users to see where the program heard each command. The program also outputs all conversation heard.

  2. Conversation heard is not stored in any place besides the terminal, and is only used for the processing of commands, rendering the program as safe for private conversation. After the program is silenced, the terminal history is wiped.
