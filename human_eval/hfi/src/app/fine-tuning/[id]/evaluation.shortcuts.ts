export type KeyboardShortcutParams = {
  save: () => void;
  nextInstance: () => void;
  nextInstanceIfSaved: () => void;
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
      condition: (key, isModified) => key === 'i' && !isModified,
      handler: () => { params.previousQuestion(); params.approve() }
    },
    {
      condition: (key, isModified) => key === 'u' && !isModified,
      handler: () => { params.previousQuestion(); params.reject() }
    },
    {
      condition: (key, isModified) => key === 'k' && !isModified,
      handler: () => { params.nextQuestion(); params.approve() }
    },
    {
      condition: (key, isModified) => key === 'j' && !isModified,
      handler: () => { params.nextQuestion(); params.reject() }
    },
    {
      condition: (key, _) => ['1', '2', '3', '4', '5'].includes(key),
      handler: (e) => params.rank(Number(e.key))
    },
    {
      condition: (key, _) => key === ' ',
      handler: (e) => { e.preventDefault(); params.nextInstanceIfSaved() }
    }
  ]

  return (e: KeyboardEvent) => {

    const allowedKeys = ['j', 'k', '1', '2', '3', '4', '5', 's', 'i', 'u', 'k', 'j', ' ']

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