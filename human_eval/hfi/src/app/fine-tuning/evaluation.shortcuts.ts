export type KeyboardShortcutParams = {
  save: () => void;
  nextInstance: () => void;
  previousInstance: () => void;
  nextQuestion: () => void;
  previousQuestion: () => void;
  rank: (rank: number) => void;
  approve: () => void;
  reject: () => void;
}

type KeyboardListener = (e: KeyboardEvent)  => any;

type KeyListener = {
  condition: (key: string, isModified: boolean) => boolean
  handler: (e: KeyboardEvent) => void
}

export function HandleKeyboardShortcut(params: KeyboardShortcutParams): KeyboardListener {
  
  const keyListeners: KeyListener[] = [
    {
      condition: (key, isModified) => key === 's' && isModified,
      handler: (e) => { e.preventDefault(); params.save() }
    },
    {
      condition: (key, isModified) => key === 'j' && isModified,
      handler: (e) => { e.preventDefault(); params.nextInstance() }
    },
    {
      condition: (key, isModified) => key === 'k' && isModified,
      handler: (e) => { e.preventDefault(); params.previousInstance() }
    },
    {
      condition: (key, isModified) => key === 'j' && !isModified,
      handler: () => params.nextQuestion()
    },
    {
      condition: (key, isModified) => key === 'k' && !isModified,
      handler: () => params.previousQuestion()
    },
    {
      condition: (key, isModified) => key === 't' && !isModified,
      handler: () => { params.previousQuestion(); params.approve() }
    },
    {
      condition: (key, isModified) => key === 'r' && !isModified,
      handler: () => { params.previousQuestion(); params.reject() }
    },
    {
      condition: (key, isModified) => key === 'g' && !isModified,
      handler: () => { params.nextQuestion(); params.approve() }
    },
    {
      condition: (key, isModified) => key === 'f' && !isModified,
      handler: () => { params.nextQuestion(); params.reject() }
    },
    {
      condition: (key, _) => ['1', '2', '3', '4', '5'].includes(key),
      handler: (e) => params.rank(Number(e.key))
    }
  ]

  return (e: KeyboardEvent) => {

    const allowedKeys = ['j', 'k', 'r', 't', '1', '2', '3', '4', '5', 's', 'f', 'g']
    
    if (!allowedKeys.includes(e.key)) {
      return
    }
  
    const isMetaKey = e.metaKey;
    const isCtrlKey = e.ctrlKey;
  
    const isModified = isMetaKey || isCtrlKey;

    keyListeners
      .filter((l) => l.condition(e.key, isModified))
      .forEach((l) => l.handler(e));

  }
}