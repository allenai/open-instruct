import ChatMessage from "../../components/chat-message"

export type InstructionAndInputParams = {
  prompt: string|undefined
  description: string|undefined
}

export default function InstructionAndInput({ prompt, description }: InstructionAndInputParams) {
  return (
    <div id="history-message-region" className="flex m-4 p-4 rounded w-full">
        <div className="flex flex-col sm:flex-row h-fit w-full">
          <ChatMessage icon="🧑" message={prompt ?? 'loading prompt...'}>
            <p>{description}</p>
            </ChatMessage>
        </div>
    </div>
  )
}