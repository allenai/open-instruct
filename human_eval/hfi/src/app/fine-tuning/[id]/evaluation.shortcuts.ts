export type KeyboardShortcutParams = {
  save: () => void;
  nextInstance: () => void;
  nextInstanceIfSaved: () => void;
  previousInstance: () => void;
  nextQuestion: () => void;
  previousQuestion: () => void;
  rank: (rank: string) => void;
  approve: (q: number) => void;
  reject: (q: number) => void;
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
      condition: (key, isModified) => key === 'u' && !isModified,
      handler: () => { params.previousQuestion(); params.approve(1) }
    },
    {
      condition: (key, isModified) => key === 'i' && !isModified,
      handler: () => { params.previousQuestion(); params.reject(1) }
    },
    {
      condition: (key, isModified) => key === 'j' && !isModified,
      handler: () => { params.nextQuestion(); params.approve(2) }
    },
    {
      condition: (key, isModified) => key === 'k' && !isModified,
      handler: () => { params.nextQuestion(); params.reject(2) }
    },
    {
      condition: (key, _) => ['1', '2', '3', '4', '5'].includes(key),
      handler: (e) => {
        [
          { label: "A is clearly better", value: "a-is-better" },
          { label: "A is slightly better", value: "a-is-slightly-better" },
          { label: "Tie", value: "tie" },
          { label: "B is slightly better", value: "b-is-slightly-better" },
          { label: "B is clearly better", value: "b-is-better" }
        ]

        switch (e.key) {
          case '1': params.rank('a-is-better'); break;
          case '2': params.rank('a-is-slightly-better'); break;
          case '3': params.rank('tie'); break;
          case '4': params.rank('b-is-slightly-better');  break;
          case '5': params.rank('b-is-better'); break;
          
        }
      }
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