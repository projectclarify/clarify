import { html, fixture, expect } from '@open-wc/testing';

import '../view-account.js';

describe('ViewAccount', () => {
  it('has a default title "Hey there" and counter 5', async () => {
    const el = await fixture(html`
      <view-account></view-account>
    `);

    expect(el.title).to.equal('Hey there');
    expect(el.counter).to.equal(5);
  });

  it('increases the counter on button click', async () => {
    const el = await fixture(html`
      <view-account></view-account>
    `);
    el.shadowRoot.querySelector('button').click();

    expect(el.counter).to.equal(6);
  });

  it('can override the title via attribute', async () => {
    const el = await fixture(html`
      <view-account title="attribute title"></view-account>
    `);

    expect(el.title).to.equal('attribute title');
  });

  it('shows initially the text "hey there Nr. 5!" and an "increment" button', async () => {
    const el = await fixture(html`
      <view-account></view-account>
    `);

    expect(el).shadowDom.to.equalSnapshot();
  });

  it('passes the a11y audit', async () => {
    const el = await fixture(html`
      <view-account></view-account>
    `);

    await expect(el).shadowDom.to.be.accessible();
  });
});
